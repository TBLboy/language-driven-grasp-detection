#!/usr/bin/env python3
"""Verify Grasp-Anything++ sample-level alignment.

For a random selection of stems, check that
  - <stem>.pkl exists under --instructions-dir and decodes to a Python str
  - <stem>.pt exists under --positive-dir and loads to a [N, 6] torch tensor

Usage:
  python research/scripts/verify_downloaded_dataset.py \
    --instructions-dir <...>/grasp_instructions \
    --positive-dir <...>/grasp_label_positive \
    [--n 5] [--seed 0]
"""

from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instructions-dir", required=True)
    parser.add_argument("--positive-dir", required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-instruction", action="store_true",
                        help="Print the decoded instruction string")
    args = parser.parse_args()

    instr_dir = Path(args.instructions_dir)
    pos_dir = Path(args.positive_dir)

    if not instr_dir.is_dir():
        print(f"[FAIL] instructions dir not found: {instr_dir}", file=sys.stderr)
        return 2
    if not pos_dir.is_dir():
        print(f"[FAIL] positive dir not found: {pos_dir}", file=sys.stderr)
        return 2

    instr_files = sorted(instr_dir.glob("*.pkl"))
    if not instr_files:
        print("[FAIL] no .pkl instruction files found", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    sample = rng.sample(instr_files, k=min(args.n, len(instr_files)))

    ok = 0
    for instr_path in sample:
        stem = instr_path.stem
        pos_path = pos_dir / f"{stem}.pt"
        print(f"stem = {stem}")
        print(f"  instruction_path = {instr_path}")
        print(f"  positive_path    = {pos_path}")

        if not pos_path.is_file():
            print("  [FAIL] positive label missing")
            continue

        try:
            instruction = pickle.loads(instr_path.read_bytes())
        except Exception as exc:
            print(f"  [FAIL] instruction pickle load: {exc}")
            continue

        try:
            positive = torch.load(pos_path, weights_only=True)
        except Exception as exc:
            print(f"  [FAIL] positive torch load: {exc}")
            continue

        if not isinstance(instruction, str):
            print(f"  [FAIL] instruction type = {type(instruction).__name__}")
            continue

        shape = tuple(positive.shape)
        if positive.dim() != 2 or shape[-1] != 6:
            print(f"  [FAIL] positive shape = {shape}, expected [N, 6]")
            continue

        ok += 1
        if args.print_instruction:
            instr_repr = instruction
        else:
            instr_repr = (instruction[:60] + "...") if len(instruction) > 60 else instruction
        print(f"  [OK] instruction = {instr_repr!r}")
        print(f"        positive.shape = {shape}, dtype = {positive.dtype}, "
              f"first_row = {positive[0].tolist()}")

    print(f"{ok}/{len(sample)} samples aligned")
    return 0 if ok == len(sample) else 1


if __name__ == "__main__":
    sys.exit(main())
