#!/usr/bin/env python3
"""Summarize Clean LGDM experiment eval_metrics.json files.

Scans a directory of experiment subdirectories and prints a compact table of
the greedy IoU success count used by the LGD evaluation protocol.

Usage:
    PYTHONNOUSERSITE=1 \
    /home/tbl/miniforge3/envs/grasp-lgd/bin/python \
      research/scripts/summarize_experiments.py outputs/lgdm_exp1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--pattern", default="eval_metrics.json")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = Path(__file__).resolve().parent.parent.parent / root
    if not root.exists():
        print(f"error: root not found: {root}", file=sys.stderr)
        return 1

    rows = []
    for metrics_path in sorted(root.rglob(args.pattern)):
        exp = str(metrics_path.parent.relative_to(root))
        data = json.loads(metrics_path.read_text())
        results = data.get("results", [])
        correct = sum(1 for r in results if r.get("correct"))
        total = data.get("num", len(results))
        rows.append(
            {
                "experiment": exp,
                "correct": correct,
                "total": total,
                "rate": (correct / total) if total else 0.0,
            }
        )

    rows.sort(key=lambda r: r["experiment"])
    print(f"{'experiment':<22}{'correct':>9}{'total':>7}{'rate':>9}")
    for r in rows:
        print(
            f"{r['experiment']:<22}{r['correct']:>9}{r['total']:>7}"
            f"{r['rate'] * 100:>8.1f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
