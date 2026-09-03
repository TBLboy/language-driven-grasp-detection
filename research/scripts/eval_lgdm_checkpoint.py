#!/usr/bin/env python3
"""Repeated 10-step evaluation for a saved LGDM / LGDM+LSAR checkpoint.

writes per-seed correct counts and a small summary JSON so single-run
diffusion sampling variance does not drive experimental conclusions.

Usage:
    PYTHONNOUSERSITE=1 \\
    /home/tbl/miniforge3/envs/grasp-lgd/bin/python \\
      research/scripts/eval_lgdm_checkpoint.py \\
      --checkpoint outputs/lgdm_scale_sweep/lsar_scale_0.01/last.pt \\
      --stems-tsv research/smoke-data/train_subset_1000.tsv \\
      --out outputs/lgdm_scale_sweep/lsar_scale_0.01/repeated_eval.json \\
      --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "LGD-main"))

from inference.models.lgdm.network import LGDM  # noqa: E402
from inference.post_process import post_process_output  # noqa: E402
from utils.dataset_processing import evaluation  # noqa: E402

from models.lgdm_lsar import LGDMWithConditioning  # noqa: E402
from research.scripts.train_lgdm_clean import (  # noqa: E402
    CleanLGDMRealDataset,
    create_respaced_diffusion,
    parse_tsv,
)


def evaluate_once(
    net,
    eval_diffusion,
    dataset,
    device,
    seed: int,
    indices: list[int] | None = None,
) -> tuple[int, int]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    net.eval()
    correct = 0
    loop_indices = indices if indices is not None else list(range(len(dataset)))
    with torch.no_grad():
        for i in loop_indices:
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
            ok = evaluation.calculate_iou_match(
                q_img,
                ang_img,
                gt_bbs,
                no_grasps=1,
                grasp_width=width_img,
                threshold=0.25,
            )
            correct += int(ok)
    return correct, len(loop_indices)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stems-tsv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="Alias for --repeats (kept for older commands).",
    )
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Evaluate only a deterministic subset of validation samples.",
    )
    parser.add_argument(
        "--subsample-seed",
        type=int,
        default=7,
        help="Random seed used to select the --max-samples subset.",
    )
    parser.add_argument("--start-seed", type=int, default=100)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Alias for --start-seed (kept for older commands).",
    )
    parser.add_argument("--condition-mode", default=None)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Override the checkpoint's train/val split seed.",
    )
    args = parser.parse_args()
    num_repeats = args.repeats if args.repeat is None else args.repeat
    start_seed = args.start_seed if args.seed is None else args.seed

    checkpoint = Path(args.checkpoint)
    checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
    stems_path = Path(args.stems_tsv)
    stems_path = stems_path if stems_path.is_absolute() else ROOT / stems_path
    out_path = Path(args.out)
    out_path = out_path if out_path.is_absolute() else ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta_path = checkpoint.parent / "args.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    condition_mode = args.condition_mode or meta.get("condition_mode", "none")
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
    split_seed = int(
        args.split_seed
        if args.split_seed is not None
        else meta.get("split_seed", meta.get("seed", 42))
    )

    rows = parse_tsv(stems_path)
    n = len(rows)
    split = int(np.floor(train_ratio * n))
    perm = torch.randperm(
        n, generator=torch.Generator().manual_seed(split_seed)
    ).tolist()
    val_rows = [rows[i] for i in perm[split:]]
    val_dataset = CleanLGDMRealDataset(
        val_rows,
        instruction_dir=instruction_dir,
        positive_dir=positive_dir,
        image_dir=image_dir,
    )
    sample_indices: list[int] | None = None
    if args.max_samples > 0:
        n = len(val_dataset)
        generator = torch.Generator().manual_seed(args.subsample_seed)
        sample_indices = torch.randperm(
            n, generator=generator
        )[: min(args.max_samples, n)].tolist()

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
    eval_diffusion = create_respaced_diffusion(args.eval_steps)

    repeats = []
    for i in range(num_repeats):
        seed = start_seed + i
        correct, total = evaluate_once(
            net,
            eval_diffusion,
            val_dataset,
            device,
            seed,
            indices=sample_indices,
        )
        repeats.append(
            {"seed": seed, "correct": correct, "total": total}
        )
        print(f"seed {seed}: {correct}/{total} correct", flush=True)

    summary = {
        "checkpoint": str(checkpoint),
        "condition_mode": condition_mode,
        "eval_steps": args.eval_steps,
        "max_samples": args.max_samples,
        "subsample_seed": args.subsample_seed,
        "repeats": repeats,
        "mean_correct": float(
            np.mean([r["correct"] for r in repeats])
        ),
        "std_correct": float(
            np.std([r["correct"] for r in repeats], ddof=1)
            if len(repeats) > 1
            else 0.0
        ),
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
