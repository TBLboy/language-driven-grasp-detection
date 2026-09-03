#!/usr/bin/env python3
"""Render a few validation samples with GT and predicted grasp rectangles.

Loads a saved Clean LGDM / LGDM+LSAR checkpoint, runs 10-step diffusion
sampling on a small number of held-out stems, and saves PNG figures that
overlay ground-truth (green) and predicted (red) grasp rectangles.

Usage:
    PYTHONNOUSERSITE=1 \\
    /home/tbl/miniforge3/envs/grasp-lgd/bin/python \\
      research/scripts/visualize_lgdm_samples.py \\
      --checkpoint outputs/lgdm_exp1000/lsar_tuned/last.pt \\
      --stems-tsv research/smoke-data/train_subset_1000.tsv \\
      --out outputs/lgdm_exp1000/visuals --n-samples 6
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stems-tsv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--condition-mode", default=None)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument(
        "--show-affordance",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Overlay the LSAR affordance map. Defaults to true for lsar mode.",
    )
    parser.add_argument(
        "--sample-stems",
        default=None,
        help="Comma-separated stems to visualize instead of first N val samples.",
    )
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
    stems_path = Path(args.stems_tsv)
    stems_path = stems_path if stems_path.is_absolute() else ROOT / stems_path
    out_dir = Path(args.out)
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = checkpoint.parent / "args.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    condition_mode = args.condition_mode or meta.get("condition_mode", "none")
    show_affordance = (
        args.show_affordance
        if args.show_affordance is not None
        else condition_mode == "lsar"
    )
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
    if condition_mode == "none":
        net = LGDM(input_channels=3).to(device)
    else:
        net = LGDMWithConditioning(
            input_channels=3, condition_mode=condition_mode
        ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    net.load_state_dict(state["model_state_dict"])
    net.eval()
    eval_diffusion = create_respaced_diffusion(args.eval_steps)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    n_samples = min(args.n_samples, len(val_rows))
    n_cols = min(3, n_samples)
    n_rows = int(np.ceil(n_samples / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False
    )
    axes = axes.ravel()
    summary = []
    with torch.no_grad():
        for i in range(n_samples):
            x, y, stem, query = val_dataset[i]
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
            aff_overlay = None
            if show_affordance and getattr(net, "affordance_map", None) is not None:
                aff_map = net.affordance_map[0, 0].detach().cpu().numpy()
                lo, hi = float(aff_map.min()), float(aff_map.max())
                if hi - lo > 1e-8:
                    aff_map = (aff_map - lo) / (hi - lo)
                else:
                    aff_map = np.zeros_like(aff_map)
                aff_overlay = resize(
                    aff_map,
                    (224, 224),
                    anti_aliasing=True,
                    preserve_range=True,
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
            gt_bbs = val_dataset.get_gtbb(stem)
            grasps = detect_grasps(
                q_img, ang_img, width_img=width_img, no_grasps=1
            )
            correct = bool(
                evaluation.calculate_iou_match(
                    q_img,
                    ang_img,
                    gt_bbs,
                    no_grasps=1,
                    grasp_width=width_img,
                    threshold=0.25,
                )
            )
            scene = stem.rsplit("_", 2)[0]
            rgb = load_rgb(image_dir / f"{scene}.jpg")
            ax = axes[i]
            ax.imshow(rgb)
            if aff_overlay is not None:
                ax.imshow(aff_overlay, cmap="jet", alpha=0.55)
            for g in gt_bbs:
                g.plot(ax, color="green")
            for g in grasps:
                g.plot(ax, color="red")
            ax.set_title(f"{stem[:12]}... {'OK' if correct else 'FAIL'}\n{query}")
            ax.axis("off")
            summary.append(
                {
                    "stem": stem,
                    "instruction": query,
                    "correct": correct,
                    "n_gt": len(gt_bbs.grs),
                    "affordance_rendered": aff_overlay is not None,
                }
            )

    fig.tight_layout()
    fig.savefig(out_dir / "qualitative.png", dpi=120)
    plt.close(fig)
    (out_dir / "summary.json").write_text(
        json.dumps({"samples": summary}, indent=2), encoding="utf-8"
    )
    print(f"saved {out_dir / 'qualitative.png'} with {len(summary)} samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
