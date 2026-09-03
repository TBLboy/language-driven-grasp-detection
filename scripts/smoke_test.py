#!/usr/bin/env python
"""Run one real Grasp-Anything++ sample through the baseline smoke chain."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "LGD-main"))

from data_utils.grasp_anything_pp import GraspAnythingPPSampleDataset
from inference.models.lgrconvnet3 import GenerativeResnet
from inference.post_process import post_process_output
from utils.dataset_processing import evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT / "research/smoke-data"))
    parser.add_argument(
        "--stem",
        default="805944ac6070b2c8f52a2ef228c9b660e116af1221284245dfa4930c8be865a6_0_1",
    )
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda")

    stems = [args.stem]
    dataset = GraspAnythingPPSampleDataset(args.root, stems)
    x, y, idx, rot, zoom, prompt, query = dataset[0]
    xb = x.unsqueeze(0).to(device)
    yb = [t.unsqueeze(0).to(device) for t in y]
    expected_map = (1, 1, 224, 224)
    assert tuple(xb.shape) == (1, 3, 224, 224)
    assert all(tuple(t.shape) == expected_map for t in yb)

    print(f"input: {tuple(xb.shape)} {xb.dtype}")
    print(f"gt maps: {[tuple(t.shape) for t in yb]}")
    print(f"instruction: {prompt!r}")

    net = GenerativeResnet(input_channels=3, dropout=False, channel_size=32).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    out = net(xb, [prompt], [query])
    assert all(tuple(t.shape) == expected_map for t in out)
    print(f"forward maps: {[tuple(t.shape) for t in out]}")

    lossd = net.compute_loss(xb, yb, [prompt], [query])
    print(f"loss: {lossd['loss'].item():.6f}")
    print(f"loss terms: { {k: v.item() for k, v in lossd['losses'].items()} }")

    optimizer.zero_grad()
    lossd["loss"].backward()
    optimizer.step()
    print("backward + optimizer step: OK")

    q_img, ang_img, width_img = post_process_output(
        lossd["pred"]["pos"],
        lossd["pred"]["cos"],
        lossd["pred"]["sin"],
        lossd["pred"]["width"],
    )
    assert q_img.shape == (224, 224)
    assert ang_img.shape == (224, 224)
    assert width_img.shape == (224, 224)
    print(f"post-process: q={q_img.shape} angle={ang_img.shape} width={width_img.shape}")

    gt_bbs = dataset.get_gtbb(idx, rot, zoom)
    correct = evaluation.calculate_iou_match(
        q_img,
        ang_img,
        gt_bbs,
        no_grasps=1,
        grasp_width=width_img,
        threshold=args.threshold,
    )
    assert isinstance(correct, bool)
    print(f"gt rectangles: {len(gt_bbs.grs)}")
    print(f"correct: {correct}")


if __name__ == "__main__":
    main()
