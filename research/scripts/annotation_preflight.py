#!/usr/bin/env python3
"""Run the annotation preflight for a fixed list of Grasp-Anything++ stems.

For each stem in ``--stems`` we verify:

* the instruction pickle exists and decodes to a non-empty ``str``
* the positive grasp ``.pt`` exists and loads to a ``[N, 6]`` float tensor
* ``N > 0``
* all values are finite

The script writes a TSV report and a short text summary; it does not touch
RGB images and never falls back to a fake image. Exit code 0 means every
stem passed.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import torch


def load_stems(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]


def check_stem(
    stem: str,
    instr_dir: Path,
    pos_dir: Path,
) -> tuple[str, dict]:
    info: dict = {"stem": stem}
    parts = stem.split("_")
    if len(parts) < 3:
        info["status"] = "FAIL-bad-stem"
        return stem, info
    info["scene"] = parts[0]
    info["obj"] = parts[1]
    info["part"] = parts[2]

    instr_path = instr_dir / f"{stem}.pkl"
    pos_path = pos_dir / f"{stem}.pt"
    info["instruction_path"] = str(instr_path)
    info["positive_path"] = str(pos_path)

    if not instr_path.is_file():
        info["status"] = "FAIL-instruction-missing"
        return stem, info
    if not pos_path.is_file():
        info["status"] = "FAIL-positive-missing"
        return stem, info

    try:
        instruction = pickle.loads(instr_path.read_bytes())
    except Exception as exc:
        info["status"] = f"FAIL-instruction-load:{type(exc).__name__}"
        return stem, info
    if not isinstance(instruction, str) or not instruction.strip():
        info["status"] = "FAIL-instruction-empty"
        return stem, info
    info["instruction_len"] = len(instruction)
    info["instruction_preview"] = (instruction[:80] + "...") if len(instruction) > 80 else instruction

    try:
        pos = torch.load(pos_path, weights_only=True)
    except Exception as exc:
        info["status"] = f"FAIL-positive-load:{type(exc).__name__}"
        return stem, info

    if pos.dim() != 2 or tuple(pos.shape)[-1] != 6:
        info["status"] = f"FAIL-positive-shape:{tuple(pos.shape)}"
        return stem, info
    n = int(pos.shape[0])
    if n == 0:
        info["status"] = "FAIL-positive-empty"
        return stem, info
    if not torch.isfinite(pos).all().item():
        info["status"] = "FAIL-positive-nonfinite"
        return stem, info

    info["pos_shape"] = "x".join(map(str, pos.shape))
    info["pos_min"] = float(pos.min().item())
    info["pos_max"] = float(pos.max().item())
    info["first_row"] = pos[0].tolist()
    info["status"] = "OK"
    return stem, info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stems", required=True, help="Path to stems.txt")
    parser.add_argument(
        "--instructions-dir",
        default="/mnt/data/grasp-anything-lgd/data/processed/grasp-anything-pp/grasp_instructions/grasp_instructions",
    )
    parser.add_argument(
        "--positive-dir",
        default="/mnt/data/grasp-anything-lgd/data/processed/grasp-anything-pp/grasp_label_positive/grasp_label_positive",
    )
    parser.add_argument("--report", default="outputs/annotation_preflight/report.tsv")
    parser.add_argument("--summary", default="outputs/annotation_preflight/summary.txt")
    args = parser.parse_args()

    stems_path = Path(args.stems)
    instr_dir = Path(args.instructions_dir)
    pos_dir = Path(args.positive_dir)
    report_path = Path(args.report)
    summary_path = Path(args.summary)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    stems = load_stems(stems_path)
    if not stems:
        print(f"[FAIL] no stems loaded from {stems_path}", file=sys.stderr)
        return 2

    print(f"stems: {len(stems)} from {stems_path}")
    results: list[dict] = []
    for stem in stems:
        _, info = check_stem(stem, instr_dir, pos_dir)
        results.append(info)
        status = info["status"]
        print(f"  stem = {stem} -> {status}")

    fields = [
        "stem", "scene", "obj", "part", "status",
        "pos_shape", "pos_min", "pos_max",
        "instruction_len", "instruction_preview",
        "instruction_path", "positive_path",
    ]
    with report_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(fields) + "\n")
        for info in results:
            row = [str(info.get(k, "")) for k in fields]
            f.write("\t".join(row) + "\n")
    print(f"wrote {report_path}")

    ok = sum(1 for r in results if r["status"] == "OK")
    fail = len(results) - ok
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"total: {len(results)}\nok: {ok}\nfail: {fail}\n")
        for r in results:
            f.write(f"  {r['stem']}\t{r['status']}\n")
    print(f"wrote {summary_path}")
    print(f"summary: {ok}/{len(results)} OK")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
