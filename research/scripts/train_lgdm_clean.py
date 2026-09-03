#!/usr/bin/env python3
"""Train the official LGDM diffusion baseline with a clean objective.

The upstream `train_network_diffusion.py` computes the diffusion loss but
comments out its backward call, so its optimizer only updates the dense-map
loss. This script keeps the LGDM architecture and evaluation unchanged but
turns the diffusion objective back on:

    total = diffusion_mse + 1e-3 * diffusion_contrastive
          + MSE(cos) + MSE(sin) + MSE(width)
          + lambda_aff * MSE(LSAR affordance map, downsampled pos GT)

The diffusion MSE already supervises the grasp-position map, so the dense
position MSE is intentionally not duplicated.

Usage:
    PYTHONNOUSERSITE=1 \
    /home/tbl/miniforge3/envs/grasp-lgd/bin/python \
      research/scripts/train_lgdm_clean.py \
      --stems-tsv research/smoke-data/train_subset_100.tsv
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from skimage.transform import resize

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "LGD-main"))

from diffusion import gaussian_diffusion as gd  # noqa: E402
from diffusion.resample import create_named_schedule_sampler  # noqa: E402
from diffusion.respace import SpacedDiffusion, space_timesteps  # noqa: E402
from inference.models.lgdm.network import LGDM  # noqa: E402
from inference.post_process import post_process_output  # noqa: E402
from utils.dataset_processing import evaluation  # noqa: E402
from utils.dataset_processing.grasp import GraspRectangles  # noqa: E402
from utils.model_util import create_diffusion  # noqa: E402

from models.lgdm_lsar import LGDMWithConditioning  # noqa: E402


def parse_tsv(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            values = line.split("\t")
            rows.append(dict(zip(header, values)))
    return rows


class CleanLGDMRealDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        stems: list[dict],
        instruction_dir: Path,
        positive_dir: Path,
        image_dir: Path,
        output_size: int = 224,
    ) -> None:
        self.rows = stems
        self.instruction_dir = Path(instruction_dir)
        self.positive_dir = Path(positive_dir)
        self.image_dir = Path(image_dir)
        self.output_size = output_size
        self._cache: dict[int, tuple] = {}

    def _instruction(self, stem: str) -> str:
        return pickle.loads((self.instruction_dir / f"{stem}.pkl").read_bytes())

    def _image(self, stem: str) -> np.ndarray:
        scene = stem.rsplit("_", 2)[0]
        path = self.image_dir / f"{scene}.jpg"
        with PILImage.open(path) as img:
            rgb = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        rgb = resize(
            rgb,
            (self.output_size, self.output_size),
            anti_aliasing=True,
            preserve_range=False,
        )
        rgb = rgb.astype(np.float32)
        rgb -= rgb.mean(axis=(0, 1), keepdims=True)
        return np.transpose(rgb, (2, 0, 1))

    def get_gtbb(self, stem: str):
        return GraspRectangles.load_from_grasp_anything_file(
            self.positive_dir / f"{stem}.pt", scale=self.output_size / 416.0
        )

    def dense_maps(self, stem: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bbs = self.get_gtbb(stem)
        pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        width_img = np.clip(width_img, 0.0, self.output_size / 2) / (
            self.output_size / 2
        )
        return (
            pos_img.astype(np.float32),
            ang_img.astype(np.float32),
            width_img.astype(np.float32),
        )

    def __getitem__(self, idx: int):
        if idx in self._cache:
            return self._cache[idx]
        row = self.rows[idx]
        stem = row["stem"]
        x = torch.from_numpy(self._image(stem))
        pos, ang, width = self.dense_maps(stem)
        y = (
            torch.from_numpy(np.expand_dims(pos, 0).astype(np.float32)),
            torch.from_numpy(np.expand_dims(np.cos(2 * ang), 0).astype(np.float32)),
            torch.from_numpy(np.expand_dims(np.sin(2 * ang), 0).astype(np.float32)),
            torch.from_numpy(np.expand_dims(width, 0).astype(np.float32)),
        )
        instruction = self._instruction(stem)
        item = (x, y, stem, instruction)
        self._cache[idx] = item
        return item

    def __len__(self) -> int:
        return len(self.rows)


def collate_batch(batch):
    xs = torch.stack([item[0] for item in batch], dim=0)
    ys = tuple(
        torch.stack([item[1][k] for item in batch], dim=0) for k in range(4)
    )
    stems = [item[2] for item in batch]
    queries = [item[3] for item in batch]
    return xs, ys, stems, queries


def create_respaced_diffusion(sample_steps: int) -> SpacedDiffusion:
    betas = gd.get_named_beta_schedule("cosine", 1000)
    return SpacedDiffusion(
        use_timesteps=space_timesteps(1000, [sample_steps]),
        betas=betas,
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=gd.ModelVarType.FIXED_SMALL,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
    )


def train_epoch(
    net,
    diffusion,
    schedule_sampler,
    loader,
    optimizer,
    device,
    epoch,
    grad_clip,
    log_every,
    max_batches,
    lsar_affordance_weight=0.0,
):
    net.train()
    totals = []
    terms = {
        "diffusion_mse": [],
        "diffusion_contr": [],
        "diffusion_loss": [],
        "dense_cos": [],
        "dense_sin": [],
        "dense_width": [],
        "dense_pos": [],
        "affordance": [],
        "total": [],
    }
    batch_idx = 0
    start = time.time()
    for xb, yb, _stems, queries in loader:
        batch_idx += 1
        if max_batches and batch_idx > max_batches:
            break
        B = xb.shape[0]
        xb = xb.to(device)
        yb = [t.to(device) for t in yb]
        pos_gt = yb[0]
        alpha = 0.4
        idx = torch.zeros(B, dtype=torch.long, device=device)
        t, weights = schedule_sampler.sample(B, device)

        losses = diffusion.training_losses(
            net,
            pos_gt,
            xb,
            t,
            queries,
            alpha,
            idx,
        )
        diffusion_loss = (losses["loss"] * weights).mean()
        dense = net.compute_loss(
            yb,
            net.pos_output_str,
            net.cos_output_str,
            net.sin_output_str,
            net.width_output_str,
        )
        dense_aux = (
            dense["losses"]["cos_loss"]
            + dense["losses"]["sin_loss"]
            + dense["losses"]["width_loss"]
        )
        total = diffusion_loss + dense_aux
        affordance_loss = torch.zeros((), device=device)
        if (
            lsar_affordance_weight > 0
            and getattr(net, "affordance_map", None) is not None
        ):
            affordance_target = F.adaptive_avg_pool2d(pos_gt, 19)
            affordance_loss = F.mse_loss(net.affordance_map, affordance_target)
            total = total + lsar_affordance_weight * affordance_loss

        optimizer.zero_grad()
        total.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in net.parameters() if p.requires_grad],
                grad_clip,
            )
        optimizer.step()

        with torch.no_grad():
            totals.append(float(total.item()))
            terms["diffusion_mse"].append(float(losses["mse"].mean().item()))
            terms["diffusion_contr"].append(float(losses["contr"].mean().item()))
            terms["diffusion_loss"].append(float(diffusion_loss.item()))
            terms["dense_cos"].append(float(dense["losses"]["cos_loss"].item()))
            terms["dense_sin"].append(float(dense["losses"]["sin_loss"].item()))
            terms["dense_width"].append(float(dense["losses"]["width_loss"].item()))
            terms["dense_pos"].append(float(dense["losses"]["p_loss"].item()))
            terms["affordance"].append(float(affordance_loss.item()))
            terms["total"].append(float(total.item()))

        if batch_idx % log_every == 0:
            print(
                f"[epoch {epoch} batch {batch_idx}] "
                f"total={total.item():.4f} diff={diffusion_loss.item():.4f} "
                f"mse={losses['mse'].mean().item():.4f} "
                f"contr={losses['contr'].mean().item():.4f} "
                f"cos={dense['losses']['cos_loss'].item():.4f} "
                f"sin={dense['losses']['sin_loss'].item():.4f} "
                f"width={dense['losses']['width_loss'].item():.4f}",
                flush=True,
            )
            if lsar_affordance_weight > 0:
                print(
                    f"  affordance={affordance_loss.item():.4f} "
                    f"lambda={lsar_affordance_weight:.4f}",
                    flush=True,
                )

    elapsed = time.time() - start
    summary = {k: float(np.mean(v)) for k, v in terms.items()}
    summary["batches"] = batch_idx
    summary["elapsed_seconds"] = round(elapsed, 2)
    summary["batches_per_second"] = round(batch_idx / max(elapsed, 1e-6), 3)
    return summary


def evaluate(net, eval_diffusion, dataset, device, save_dir):
    net.eval()
    results = []
    with torch.no_grad():
        for i in range(len(dataset)):
            x, y, stem, query = dataset[i]
            xb = x.unsqueeze(0).to(device)
            yb = [t.unsqueeze(0).to(device) for t in y]
            pos_gt = yb[0]
            alpha = 0.4
            idx = torch.zeros(1, dtype=torch.long, device=device)

            sample = eval_diffusion.p_sample_loop(
                net,
                pos_gt.shape,
                pos_gt,
                xb,
                [query],
                alpha,
                idx,
            )
            final = net.compute_loss(
                yb,
                sample,
                net.cos_output_str,
                net.sin_output_str,
                net.width_output_str,
            )
            q_img, ang_img, width_img = post_process_output(
                final["pred"]["pos"],
                final["pred"]["cos"],
                final["pred"]["sin"],
                final["pred"]["width"],
            )
            gt_bbs = dataset.get_gtbb(stem)
            correct = evaluation.calculate_iou_match(
                q_img,
                ang_img,
                gt_bbs,
                no_grasps=1,
                grasp_width=width_img,
                threshold=0.25,
            )
            results.append(
                {
                    "stem": stem,
                    "instruction": query,
                    "correct": bool(correct),
                    "sample_shape": list(sample.shape),
                }
            )

    metrics_path = Path(save_dir) / "eval_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"results": results, "num": len(results)}, f, indent=2)
    correct = sum(1 for r in results if r["correct"])
    return correct, len(results)


def save_checkpoint(path: Path, net, optimizer, epoch, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stems-tsv", required=True)
    parser.add_argument(
        "--instruction-dir",
        default="/mnt/data/grasp-anything-lgd/data/processed/grasp-anything-pp/grasp_instructions/grasp_instructions",
    )
    parser.add_argument(
        "--positive-dir",
        default="/mnt/data/grasp-anything-lgd/data/processed/grasp-anything-pp/grasp_label_positive/grasp_label_positive",
    )
    parser.add_argument(
        "--image-dir",
        default="/mnt/data/grasp-anything-lgd/data/processed/grasp-anything/images",
    )
    parser.add_argument("--out", default="outputs/train_lgdm_clean")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-batches-per-epoch", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--condition-mode",
        choices=["none", "plain-y", "lsar"],
        default="none",
        help="none=official LGDM, plain-y=raw y injection ablation, lsar=proposed module",
    )
    parser.add_argument(
        "--lsar-affordance-weight",
        type=float,
        default=0.1,
        help="Aux MSE weight for LSAR affordance map vs downsampled pos GT",
    )
    parser.add_argument(
        "--lsar-scale",
        type=float,
        default=0.1,
        help="Initial (or fixed) LSAR residual scale",
    )
    parser.add_argument(
        "--lsar-fixed-scale",
        action="store_true",
        help="Freeze LSAR residual scale at --lsar-scale",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Seed for the train/val split. Defaults to --seed.",
    )
    args = parser.parse_args()
    if args.split_seed is None:
        args.split_seed = args.seed

    stems_tsv = Path(args.stems_tsv)
    if not stems_tsv.is_absolute():
        stems_tsv = ROOT / stems_tsv
    rows = parse_tsv(stems_tsv)
    resume_path = None
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = ROOT / resume_path

    # LGDM._init_albef resolves its ALBEF config with a repo-relative path.
    os.chdir(ROOT / "LGD-main")
    if len(rows) < 2:
        print(f"[FAIL] need at least 2 stems, got {len(rows)}")
        return 2

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = (
        torch.device("cpu")
        if args.cpu or not torch.cuda.is_available()
        else torch.device("cuda")
    )
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(rows)
    split = int(np.floor(args.train_ratio * n))
    perm = torch.randperm(
        n, generator=torch.Generator().manual_seed(args.split_seed)
    ).tolist()
    train_rows = [rows[i] for i in perm[:split]]
    val_rows = [rows[i] for i in perm[split:]]
    print(f"device: {device}")
    print(
        f"stems: {n}, train: {len(train_rows)}, val: {len(val_rows)}, "
        f"split_seed={args.split_seed}, train_seed={args.seed}"
    )

    train_dataset = CleanLGDMRealDataset(
        train_rows,
        instruction_dir=Path(args.instruction_dir),
        positive_dir=Path(args.positive_dir),
        image_dir=Path(args.image_dir),
    )
    val_dataset = CleanLGDMRealDataset(
        val_rows,
        instruction_dir=Path(args.instruction_dir),
        positive_dir=Path(args.positive_dir),
        image_dir=Path(args.image_dir),
    )
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        drop_last=False,
    )

    full_diffusion = create_diffusion()
    schedule_sampler = create_named_schedule_sampler("uniform", full_diffusion)
    eval_diffusion = None
    if args.eval_steps > 0:
        eval_diffusion = create_respaced_diffusion(args.eval_steps)

    print(f"instantiating LGDM with condition_mode={args.condition_mode}...")
    if args.condition_mode == "none":
        net = LGDM(input_channels=3).to(device)
    else:
        net = LGDMWithConditioning(
            input_channels=3,
            condition_mode=args.condition_mode,
            lsar_scale=args.lsar_scale,
            lsar_scale_trainable=not args.lsar_fixed_scale,
        ).to(device)
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"LGDM params total={total_params / 1e6:.1f}M trainable={trainable / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    start_epoch = 0
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device)
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"resumed from {resume_path}, continuing at epoch {start_epoch}")

    log_path = out_dir / "training_log.jsonl"
    history = []
    for epoch in range(start_epoch, args.epochs):
        print(f"epoch {epoch} start", flush=True)
        summary = train_epoch(
            net=net,
            diffusion=full_diffusion,
            schedule_sampler=schedule_sampler,
            loader=loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            grad_clip=args.grad_clip,
            log_every=args.log_every,
            max_batches=args.max_batches_per_epoch,
            lsar_affordance_weight=args.lsar_affordance_weight,
        )
        summary["epoch"] = epoch
        history.append(summary)
        print(
            f"epoch {epoch} done: total={summary['total']:.4f} "
            f"elapsed={summary['elapsed_seconds']:.1f}s "
            f"b/s={summary['batches_per_second']:.3f}",
            flush=True,
        )
        checkpoint = out_dir / "last.pt"
        save_checkpoint(checkpoint, net, optimizer, epoch, args)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")

    if args.eval_steps > 0:
        print("running post-training evaluation...", flush=True)
        correct, total = evaluate(net, eval_diffusion, val_dataset, device, out_dir)
        print(f"eval with {args.eval_steps} sampling steps: {correct}/{total} correct")
    else:
        print("evaluation skipped (--eval-steps 0)")
    with (out_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
