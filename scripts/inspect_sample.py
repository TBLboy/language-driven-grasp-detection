#!/usr/bin/env python
"""Inspect one local Grasp-Anything++ smoke sample."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch


def describe(path: Path):
    if not path.exists():
        return None
    data = path.read_bytes()
    if path.suffix == ".pkl":
        obj = pickle.loads(data)
        return {"type": type(obj).__name__, "value": obj}
    if path.suffix == ".pt":
        tensor = torch.load(path, weights_only=True)
        return {
            "type": type(tensor).__name__,
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "first_row": tensor[0].tolist(),
        }
    if path.suffix == ".npy":
        arr = np.load(path)
        return {
            "type": type(arr).__name__,
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    if path.suffix in {".jpg", ".jpeg", ".png"}:
        from PIL import Image

        with Image.open(path) as img:
            return {
                "type": "PIL.Image",
                "size": img.size,
                "mode": img.mode,
                "bytes": path.stat().st_size,
            }
    return {"type": "unknown", "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="research/smoke-data")
    parser.add_argument(
        "--stem",
        default="805944ac6070b2c8f52a2ef228c9b660e116af1221284245dfa4930c8be865a6_0_1",
    )
    args = parser.parse_args()

    root = Path(args.root)
    candidates = {
        "image": root / "image" / f"{args.stem.split('_')[0]}.jpg",
        "instruction": root / "grasp_instructions" / f"{args.stem}.pkl",
        "positive": root / "grasp_label_positive" / f"{args.stem}.pt",
        "negative": root / "grasp_label_negative" / f"{args.stem}.pt",
        "part_mask": root / "part_mask" / f"{args.stem}.npy",
    }

    print(f"stem: {args.stem}")
    for name, path in candidates.items():
        print(f"{name}: {path}")
        print("  ", describe(path))


if __name__ == "__main__":
    main()
