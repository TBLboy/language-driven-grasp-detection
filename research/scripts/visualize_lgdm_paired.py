#!/usr/bin/env python3
"""Render paired qualitative samples for the LGDM vs LSAR paper figure.

For each selected validation stem, the script runs 10-step diffusion sampling
with both the official LGDM baseline and the LSAR-conditioned model, then
renders:

    RGB + GT | LGDM prediction | LSAR prediction | LSAR affordance

Usage:
    PYTHONNOUSERSITE=1 \\
    /home/tbl/miniforge3/envs/grasp-lgd/bin/python \\
      research/scripts/visualize_lgdm_paired.py \\
      --baseline-checkpoint outputs/lgdm_10k/none/last.pt \\
      --lsar-checkpoint outputs/lgdm_10k/lsar_full/last.pt \\
      --stems-tsv research/smoke-data/train_subset_10k.tsv \\
      --out outputs/lgdm_10k/paired_visuals \\
      --n-samples 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image as PILImage
from skimage.transform import resize

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "LGD-main"))

from inference.models.lgdm.network import LGDM  # noqa: E402
from inference.post_process import post_process_output  # noqa: E402
from utils.dataset_processing import evaluation  # noqa: E402
from utils.dataset_processing.grasp import GraspRectangles, detect_grasps  # noqa: E402

from models.lgdm_lsar import LGDMWithConditioning  # noqa: E402
from research.scripts.train_lgdm_clean import (  # noqa: E402
    CleanLGDMRealDataset,
    create_respaced_diffusion,
    parse_tsv,
)


def load_rgb(path: Path, size: int = 224) -> np.ndarray:
    with PILImage.open(path) as img:
        rgb = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    return resize(rgb, (size, size), anti_aliasing=True, preserve_range=False)


def load_meta(checkpoint: Path) -> dict:
    meta_path = checkpoint.parent / "args.json"
    return json.loads(meta_path.read_text()) if meta_path.exists() else {}


def load_model(checkpoint: Path, condition_mode: str, device):
    if condition_mode == "none":
        net = LGDM(input_channels=3).to(device)
    else:
        net = LGDMWithConditioning(
            input_channels=3,
            condition_mode=condition_mode,
            lsar_scale=0.01,
            lsar_scale_trainable=False,
        ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    net.load_state_dict(state["model_state_dict"])
    net.eval()
    return net


def normalize_map(values: np.ndarray) -> np.ndarray:
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo > 1e-8:
        return (values - lo) / (hi - lo)
    return np.zeros_like(values)


def sample_grasp(
    net,
    diffusion,
    device,
    xb,
    yb,
    query,
):
    pos_gt = yb[0]
    alpha = 0.4
    idx = torch.zeros(1, dtype=torch.long, device=device)
    sample = diffusion.p_sample_loop(
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
    return q_img, ang_img, width_img


def evaluate_grasp(q_img, ang_img, width_img, gt_bbs):
    return bool(
        evaluation.calculate_iou_match(
            q_img,
            ang_img,
            gt_bbs,
            no_grasps=1,
            grasp_width=width_img,
            threshold=0.25,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--lsar-checkpoint", required=True)
    parser.add_argument("--stems-tsv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument(
        "--sample-stems",
        default=None,
        help="Comma-separated stems to visualize instead of first N val samples.",
    )
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    baseline_checkpoint = Path(args.baseline_checkpoint)
    baseline_checkpoint = (
        baseline_checkpoint
        if baseline_checkpoint.is_absolute()
        else ROOT / baseline_checkpoint
    )
    lsar_checkpoint = Path(args.lsar_checkpoint)
    lsar_checkpoint = (
        lsar_checkpoint if lsar_checkpoint.is_absolute() else ROOT / lsar_checkpoint
    )
    stems_path = Path(args.stems_tsv)
    stems_path = stems_path if stems_path.is_absolute() else ROOT / stems_path
    out_dir = Path(args.out)
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(lsar_checkpoint)
    instruction_dir = Path(
        meta.get(
            "instruction_dir",
            "/mnt/data/grasp-anything-lgd/data/processed/grasp-anything-pp/"
            "grasp_instructions/grasp_instructions",
        )
    )
    positive_dir = Path(
        meta.get(
            "positive_dir",
            "/mnt/data/grasp-anything-lgd/data/processed/grasp-anything-pp/"
            "grasp_label_positive/grasp_label_positive",
        )
    )
    image_dir = Path(
        meta.get(
            "image_dir",
            "/mnt/data/grasp-anything-lgd/data/processed/grasp-anything/images",
        )
    )
    train_ratio = float(meta.get("train_ratio", 0.8))
    split_seed = int(meta.get("seed", 42))

    rows = parse_tsv(stems_path)
    n = len(rows)
    split = int(np.floor(train_ratio * n))
    perm = torch.randperm(
        n, generator=torch.Generator().manual_seed(split_seed)
    ).tolist()
    val_rows = [rows[i] for i in perm[split:]]
    if args.sample_stems:
        selected = {
            stem.strip() for stem in args.sample_stems.split(",") if stem.strip()
        }
        val_rows = [row for row in val_rows if row["stem"] in selected]
        if not val_rows:
            raise SystemExit(
                "none of --sample-stems matched the validation split; "
                "check the stem list"
            )
    val_dataset = CleanLGDMRealDataset(
        val_rows,
        instruction_dir=instruction_dir,
        positive_dir=positive_dir,
        image_dir=image_dir,
    )

    os.chdir(ROOT / "LGD-main")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_net = load_model(baseline_checkpoint, "none", device)
    lsar_net = load_model(lsar_checkpoint, "lsar", device)
    eval_diffusion = create_respaced_diffusion(args.eval_steps)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    n_samples = min(args.n_samples, len(val_rows))
    columns = ["RGB + GT", "LGDM baseline", "LSAR (ours)", "LSAR affordance"]
    fig, axes = plt.subplots(
        n_samples,
        len(columns),
        figsize=(20, 5 * n_samples),
        squeeze=False,
    )
    for j, label in enumerate(columns):
        axes[0][j].set_title(label, fontsize=12)

    summary = []
    with torch.no_grad():
        for i in range(n_samples):
            x, y, stem, query = val_dataset[i]
            xb = x.unsqueeze(0).to(device)
            yb = [t.unsqueeze(0).to(device) for t in y]
            gt_bbs = val_dataset.get_gtbb(stem)

            base_q, base_ang, base_width = sample_grasp(
                baseline_net,
                eval_diffusion,
                device,
                xb,
                yb,
                query,
            )
            lsar_q, lsar_ang, lsar_width = sample_grasp(
                lsar_net,
                eval_diffusion,
                device,
                xb,
                yb,
                query,
            )
            base_grasps = detect_grasps(
                base_q,
                base_ang,
                width_img=base_width,
                no_grasps=1,
            )
            lsar_grasps = detect_grasps(
                lsar_q,
                lsar_ang,
                width_img=lsar_width,
                no_grasps=1,
            )
            base_ok = evaluate_grasp(base_q, base_ang, base_width, gt_bbs)
            lsar_ok = evaluate_grasp(lsar_q, lsar_ang, lsar_width, gt_bbs)

            aff_overlay = None
            if getattr(lsar_net, "affordance_map", None) is not None:
                aff_map = lsar_net.affordance_map[0, 0].detach().cpu().numpy()
                aff_overlay = resize(
                    normalize_map(aff_map),
                    (224, 224),
                    anti_aliasing=True,
                    preserve_range=True,
                )

            scene = stem.rsplit("_", 2)[0]
            rgb = load_rgb(image_dir / f"{scene}.jpg")

            axes[i][0].imshow(rgb)
            for g in gt_bbs:
                g.plot(axes[i][0], color="green")
            axes[i][1].imshow(rgb)
            for g in base_grasps:
                g.plot(axes[i][1], color="red")
            axes[i][2].imshow(rgb)
            for g in lsar_grasps:
                g.plot(axes[i][2], color="red")
            axes[i][3].imshow(rgb)
            if aff_overlay is not None:
                axes[i][3].imshow(aff_overlay, cmap="jet", alpha=0.55)

            axes[i][0].set_ylabel(
                f"{stem[:14]}...\nBASE={'OK' if base_ok else 'FAIL'}\n"
                f"LSAR={'OK' if lsar_ok else 'FAIL'}",
                fontsize=9,
            )
            axes[i][0].set_title(
                f"{stem[:12]}... | {query}",
                fontsize=9,
            )
            for j in range(len(columns)):
                axes[i][j].axis("off")

            summary.append(
                {
                    "stem": stem,
                    "instruction": query,
                    "baseline_correct": base_ok,
                    "lsar_correct": lsar_ok,
                    "n_gt": len(gt_bbs.grs),
                    "affordance_rendered": aff_overlay is not None,
                }
            )

    fig.tight_layout()
    fig.savefig(out_dir / "paired_qualitative.png", dpi=120)
    plt.close(fig)
    (out_dir / "summary.json").write_text(
        json.dumps({"samples": summary}, indent=2),
        encoding="utf-8",
    )
    print(f"saved {out_dir / 'paired_qualitative.png'} with {len(summary)} samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
