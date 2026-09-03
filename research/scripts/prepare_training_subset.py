#!/usr/bin/env python3
"""Sample a deterministic Grasp-Anything++ training subset.

The script scans only positive-grasp filenames, keeps a small per-scene
candidate pool using a deterministic hash, validates the selected stems'
instruction and positive tensor, and writes a TSV that can be consumed by
the Clean LGDM training smoke. By default it enforces one stem per scene;
`--allow-same-scene` keeps multiple stems per scene up to `--max-per-scene`.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import sys
from pathlib import Path

import torch


def parse_stem(stem: str) -> tuple[str, str, str]:
    scene, obj, part = stem.rsplit("_", 2)
    return scene, obj, part


def rank_key(seed: int, stem: str) -> str:
    """Small deterministic rank so directory order does not decide sampling."""
    return hashlib.sha256(f"{seed}:{stem}".encode("utf-8")).hexdigest()


def collect_candidates(
    positive_dir: Path,
    num_needed: int,
    max_per_scene: int,
    seed: int,
    image_dir: Path | None = None,
    scene_list: set[str] | None = None,
) -> list[str]:
    """Reservoir-scan positives without loading tensors."""
    image_scenes = (
        {path.stem for path in image_dir.glob("*.jpg")}
        if image_dir is not None
        else None
    )
    if scene_list is None:
        available_scenes = image_scenes
    elif image_scenes is None:
        available_scenes = scene_list
    else:
        available_scenes = scene_list | image_scenes
    by_scene: dict[str, list[tuple[str, str]]] = {}
    count = 0
    with os.scandir(positive_dir) as it:
        for entry in it:
            if not entry.name.endswith(".pt"):
                continue
            stem = entry.name[:-3]
            scene, _obj, _part = parse_stem(stem)
            if available_scenes is not None and scene not in available_scenes:
                continue
            rank = rank_key(seed, stem)
            bucket = by_scene.setdefault(scene, [])
            bucket.append((rank, stem))
            if len(bucket) > max_per_scene:
                bucket.sort(key=lambda x: x[0])
                del bucket[max_per_scene:]
            count += 1

    if not by_scene:
        raise RuntimeError(f"no .pt candidates found under {positive_dir}")

    ranked: list[tuple[str, str]] = []
    for bucket in by_scene.values():
        ranked.extend(bucket)
    ranked.sort(key=lambda x: x[0])

    # Keep a margin because a few annotated stems can have empty positives or
    # a corrupted instruction. The caller validates until the target is met.
    return [stem for _rank, stem in ranked[: max(num_needed * 4, 200)]]


def validate_stem(stem: str, instruction_dir: Path, positive_dir: Path) -> dict:
    instruction_path = instruction_dir / f"{stem}.pkl"
    positive_path = positive_dir / f"{stem}.pt"
    scene, obj, part = parse_stem(stem)

    try:
        instruction = pickle.loads(instruction_path.read_bytes())
    except Exception as exc:
        return {"stem": stem, "ok": False, "reason": f"instruction-load:{type(exc).__name__}"}
    if not isinstance(instruction, str) or not instruction.strip():
        return {"stem": stem, "ok": False, "reason": "instruction-not-str"}

    try:
        positive = torch.load(positive_path, map_location="cpu")
    except Exception as exc:
        return {"stem": stem, "ok": False, "reason": f"positive-load:{type(exc).__name__}"}
    if not isinstance(positive, torch.Tensor) or positive.dim() != 2 or positive.shape[1] != 6:
        return {"stem": stem, "ok": False, "reason": f"positive-shape:{tuple(positive.shape)}"}
    num = int(positive.shape[0])
    if num <= 0 or not bool(torch.isfinite(positive).all().item()):
        return {"stem": stem, "ok": False, "reason": "positive-not-finite-or-empty"}

    return {
        "stem": stem,
        "ok": True,
        "scene": scene,
        "obj": obj,
        "part": part,
        "instruction_path": str(instruction_path),
        "positive_path": str(positive_path),
        "pos_shape": f"{num}x6",
        "pos_min": round(float(positive.min().item()), 6),
        "pos_max": round(float(positive.max().item()), 6),
        "instruction_preview": instruction[:100],
        "instruction_len": len(instruction),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-dir", required=True)
    parser.add_argument("--instruction-dir", required=True)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--scene-list", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-stems", type=int, default=100)
    parser.add_argument("--max-per-scene", type=int, default=1)
    parser.add_argument(
        "--allow-same-scene",
        action="store_true",
        help="Allow multiple stems from the same scene up to --max-per-scene.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    positive_dir = Path(args.positive_dir)
    instruction_dir = Path(args.instruction_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene_set = None
    if args.scene_list:
        scene_set = {
            line.strip()
            for line in Path(args.scene_list).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        print(f"scene list: {len(scene_set)} available scenes", file=sys.stderr)

    candidates = collect_candidates(
        positive_dir=positive_dir,
        num_needed=args.num_stems,
        max_per_scene=args.max_per_scene,
        seed=args.seed,
        image_dir=Path(args.image_dir) if args.image_dir else None,
        scene_list=scene_set,
    )
    print(f"scanned candidates: {len(candidates)}", file=sys.stderr)

    rows: list[dict] = []
    seen_scenes: set[str] = set()
    for stem in candidates:
        info = validate_stem(stem, instruction_dir=instruction_dir, positive_dir=positive_dir)
        if not info["ok"]:
            print(f"skip invalid {stem}: {info['reason']}", file=sys.stderr)
            continue
        if not args.allow_same_scene:
            if info["scene"] in seen_scenes:
                continue
            seen_scenes.add(info["scene"])
        rows.append(info)
        print(
            f"[{len(rows)}/{args.num_stems}] {stem} "
            f"N={info['pos_shape']} instruction={info['instruction_preview'][:50]!r}",
            file=sys.stderr,
        )
        if len(rows) >= args.num_stems:
            break

    if len(rows) < args.num_stems:
        print(
            f"[FAIL] only {len(rows)} valid cross-scene stems found; "
            f"requested {args.num_stems}",
            file=sys.stderr,
        )
        return 1

    columns = [
        "stem",
        "scene",
        "obj",
        "part",
        "instruction_path",
        "positive_path",
        "pos_shape",
        "pos_min",
        "pos_max",
        "instruction_preview",
        "instruction_len",
    ]
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(str(row[col]) for col in columns) + "\n")

    print(f"wrote {out_path}")
    print(f"valid stems: {len(rows)}")
    print(f"unique scenes: {len(seen_scenes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
