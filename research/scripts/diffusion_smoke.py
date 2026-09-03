#!/usr/bin/env python3
"""Run the official LGDM diffusion baseline through a real-sample smoke chain.

This script intentionally does not try to tune the model. It proves the
following execution chain with real Grasp-Anything++ files:

    RGB + instruction + positive grasp labels
    -> dense pos/cos/sin/width maps
    -> LGDM forward
    -> diffusion training loss
    -> backward + optimizer step
    -> shorter respaced p_sample_loop
    -> post_process_output
    -> calculate_iou_match

The full official create_diffusion() schedule has 1000 steps; the smoke run
uses a 10-step respaced schedule for the p_sample_loop so it remains fast.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
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


def load_stems(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


class DiffusionSmokeDataset:
    def __init__(
        self,
        instr_dir: Path,
        pos_dir: Path,
        image_dir: Path,
        stems: list[str],
        output_size: int = 224,
    ) -> None:
        self.instr_dir = Path(instr_dir)
        self.pos_dir = Path(pos_dir)
        self.image_dir = Path(image_dir)
        self.stems = stems
        self.output_size = output_size

    def _instruction(self, stem: str) -> str:
        return pickle.loads((self.instr_dir / f"{stem}.pkl").read_bytes())

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
        path = self.pos_dir / f"{stem}.pt"
        return GraspRectangles.load_from_grasp_anything_file(
            path, scale=self.output_size / 416.0
        )

    def dense_maps(self, stem: str) -> dict:
        bbs = self.get_gtbb(stem)
        pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        width_img = np.clip(width_img, 0.0, self.output_size / 2) / (
            self.output_size / 2
        )
        return {
            "pos": pos_img.astype(np.float32),
            "ang": ang_img.astype(np.float32),
            "width": width_img.astype(np.float32),
            "rectangles": len(bbs.grs),
            "pos_shape": list(pos_img.shape),
            "ang_shape": list(ang_img.shape),
            "width_shape": list(width_img.shape),
        }


def create_respaced_diffusion(sample_steps: int) -> SpacedDiffusion:
    """Build a small respaced cosine diffusion for the smoke sample loop."""
    betas = gd.get_named_beta_schedule("cosine", 1000)
    return SpacedDiffusion(
        use_timesteps=space_timesteps(1000, [sample_steps]),
        betas=betas,
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=gd.ModelVarType.FIXED_SMALL,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stems",
        default=str(ROOT / "research/smoke-data/stems.txt"),
        help="Text file with one <scene>_<obj>_<part> stem per line",
    )
    parser.add_argument(
        "--instructions-dir",
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
    parser.add_argument("--out", default="outputs/diffusion_smoke")
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--max-stems", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    stems = load_stems(Path(args.stems))[: args.max_stems]
    if not stems:
        print(f"[FAIL] no stems loaded from {args.stems}", file=sys.stderr)
        return 2

    # LGDM._init_albef reads its ALBEF config with a repo-relative path.
    os.chdir(ROOT / "LGD-main")

    torch.manual_seed(args.seed)
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
    dataset = DiffusionSmokeDataset(
        instr_dir=Path(args.instructions_dir),
        pos_dir=Path(args.positive_dir),
        image_dir=Path(args.image_dir),
        stems=stems,
    )
    full_diffusion = create_diffusion()
    smoke_diffusion = create_respaced_diffusion(args.sample_steps)

    print(f"device: {device}")
    print(f"official diffusion steps: {full_diffusion.num_timesteps}")
    print(f"smoke sampling steps: {smoke_diffusion.num_timesteps}")
    print("instantiating LGDM...")
    net = LGDM(input_channels=3).to(device)
    net.train()

    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    schedule_sampler = create_named_schedule_sampler("uniform", smoke_diffusion)

    metrics: list[dict] = []
    for idx, stem in enumerate(stems):
        info: dict = {"stem": stem, "status": "pending"}
        try:
            instruction = dataset._instruction(stem)
            dense = dataset.dense_maps(stem)
            rgb_chw = dataset._image(stem)

            xb = torch.from_numpy(rgb_chw).unsqueeze(0).to(device)

            def to_map_tensor(arr: np.ndarray) -> torch.Tensor:
                return (
                    torch.from_numpy(np.expand_dims(arr, 0).astype(np.float32))
                    .unsqueeze(0)
                    .to(device)
                )

            yb = [
                to_map_tensor(dense["pos"]),
                to_map_tensor(np.cos(2 * dense["ang"])),
                to_map_tensor(np.sin(2 * dense["ang"])),
                to_map_tensor(dense["width"]),
            ]
            pos_gt = yb[0]
            query = [instruction]
            idx_t = torch.zeros(xb.shape[0], dtype=torch.long, device=device)
            alpha = 0.4

            assert tuple(xb.shape) == (1, 3, 224, 224)
            assert all(tuple(t.shape) == (1, 1, 224, 224) for t in yb)
            info["instruction"] = instruction
            info["gt_shape"] = [tuple(t.shape) for t in yb]
            info["gt_rectangles"] = dense["rectangles"]

            t, weights = schedule_sampler.sample(xb.shape[0], device)
            losses = smoke_diffusion.training_losses(
                net,
                pos_gt,
                xb,
                t,
                query,
                alpha,
                idx_t,
            )
            diffusion_loss = (losses["loss"] * weights).mean()
            info["diffusion_loss_before"] = float(diffusion_loss.item())
            info["diffusion_terms"] = {
                k: float(v.item()) for k, v in losses.items()
            }
            # train_network_diffusion.py computes this diffusion loss but does
            # not call backward() on it; the only backward/update uses the
            # dense-map loss below.
            info["official_diffusion_backward_used"] = False

            lossd = net.compute_loss(
                yb,
                net.pos_output_str,
                net.cos_output_str,
                net.sin_output_str,
                net.width_output_str,
            )
            optimizer.zero_grad()
            lossd["loss"].backward()
            optimizer.step()
            info["official_dense_loss"] = float(lossd["loss"].item())
            info["official_dense_terms"] = {
                k: float(v.item()) for k, v in lossd["losses"].items()
            }
            missing_grad = [
                name
                for name, p in net.named_parameters()
                if p.requires_grad and p.grad is None
            ]
            nonfinite_grad = [
                name
                for name, p in net.named_parameters()
                if p.requires_grad
                and p.grad is not None
                and not torch.isfinite(p.grad).all()
            ]
            info["grad_missing_trainable_count"] = len(missing_grad)
            info["grad_nonfinite_count"] = len(nonfinite_grad)
            info["backward_grad_finite"] = len(nonfinite_grad) == 0

            sample = smoke_diffusion.p_sample_loop(
                net,
                pos_gt.shape,
                pos_gt,
                xb,
                query,
                alpha,
                idx_t,
            )
            info["sample_shape"] = tuple(sample.shape)
            info["sample_finite"] = bool(torch.isfinite(sample).all().item())

            final_lossd = net.compute_loss(
                yb,
                sample,
                net.cos_output_str,
                net.sin_output_str,
                net.width_output_str,
            )
            q_img, ang_img, width_img = post_process_output(
                final_lossd["pred"]["pos"],
                final_lossd["pred"]["cos"],
                final_lossd["pred"]["sin"],
                final_lossd["pred"]["width"],
            )
            assert q_img.shape == ang_img.shape == width_img.shape == (224, 224)
            info["postprocess_shapes"] = [
                list(q_img.shape),
                list(ang_img.shape),
                list(width_img.shape),
            ]

            gt_bbs = dataset.get_gtbb(stem)
            correct = evaluation.calculate_iou_match(
                q_img,
                ang_img,
                gt_bbs,
                no_grasps=1,
                grasp_width=width_img,
                threshold=args.threshold,
            )
            info["correct"] = bool(correct)
            info["status"] = "OK"
            if device.type == "cuda":
                info["max_gpu_mb"] = round(
                    torch.cuda.max_memory_allocated(device) / (1024 * 1024), 1
                )
        except Exception as exc:
            info["status"] = f"FAIL:{type(exc).__name__}:{exc}"
            if device.type == "cuda":
                info["max_gpu_mb"] = round(
                    torch.cuda.max_memory_allocated(device) / (1024 * 1024), 1
                )
        metrics.append(info)
        print(f"[{idx}] {stem} -> {info['status']}")
        for key in (
            "diffusion_loss_before",
            "official_dense_loss",
            "sample_shape",
            "sample_finite",
            "correct",
            "max_gpu_mb",
        ):
            if key in info:
                print(f"  {key}: {info[key]}")

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "device": str(device),
                "official_diffusion_steps": full_diffusion.num_timesteps,
                "smoke_sampling_steps": smoke_diffusion.num_timesteps,
                "stems": metrics,
            },
            f,
            indent=2,
        )
    print(f"wrote {metrics_path}")

    ok = sum(1 for m in metrics if m["status"] == "OK")
    fail = len(metrics) - ok
    print(f"summary: {ok}/{len(metrics)} OK, {fail} FAIL")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
