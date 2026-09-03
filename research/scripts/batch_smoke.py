#!/usr/bin/env python3
"""Run the engineering batch smoke for a fixed list of Grasp-Anything++ stems.

The script iterates over each stem in ``--stems`` and, depending on the
presence of RGB files under ``--image-dir``:

* always loads the instruction pickle and positive grasp tensor
* always builds the dense ``pos / cos / sin / width`` maps and checks
  their shape / finiteness
* if the RGB is available, also runs the official LGD baseline forward /
  loss / backward / post-process / IoU evaluation pipeline (a single,
  randomly initialized network; the loss value and ``correct`` flag are
  not expected to be meaningful)

If the RGB for a stem is missing the script records ``SKIP-RGB`` and
keeps going. No fake RGB tensors are ever substituted in.

Outputs (under ``--out``, default ``outputs/batch_smoke``):

* ``metrics.json`` - per-stem status, dense-map shape, loss, correct flag
* ``summary.txt``  - human readable roll-up
* ``qualitative/<stem>.png`` - RGB + dense GT overlay for stems that ran
  the full pipeline (skipped otherwise)
"""

from __future__ import annotations

import argparse
import json
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

from inference.models.lgrconvnet3 import GenerativeResnet  # noqa: E402
from inference.post_process import post_process_output  # noqa: E402
from utils.dataset_processing import evaluation  # noqa: E402
from utils.dataset_processing.grasp import GraspRectangles  # noqa: E402


def load_stems(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]


class BatchDataset(torch.utils.data.Dataset):
    """Like ``GraspAnythingPPSampleDataset`` but with explicit input dirs."""

    def __init__(self, instr_dir: Path, pos_dir: Path, image_dir: Path | None,
                 stems: list[str], output_size: int = 224) -> None:
        self.instr_dir = Path(instr_dir)
        self.pos_dir = Path(pos_dir)
        self.image_dir = Path(image_dir) if image_dir else None
        self.stems = stems
        self.output_size = output_size

    def __len__(self) -> int:
        return len(self.stems)

    def _instruction(self, stem: str) -> str:
        return pickle.loads((self.instr_dir / f"{stem}.pkl").read_bytes())

    def _image(self, stem: str) -> np.ndarray:
        if self.image_dir is None:
            raise FileNotFoundError("image_dir not configured")
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

    def get_gtbb(self, idx: int, rot: float = 0.0, zoom: float = 1.0):
        stem = self.stems[idx]
        path = self.pos_dir / f"{stem}.pt"
        bbs = GraspRectangles.load_from_grasp_anything_file(
            path, scale=self.output_size / 416.0
        )
        if rot:
            bbs.rotate(rot, (self.output_size // 2, self.output_size // 2))
        if zoom != 1.0:
            bbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        return bbs

    def has_image(self, stem: str) -> bool:
        if self.image_dir is None:
            return False
        scene = stem.rsplit("_", 2)[0]
        return (self.image_dir / f"{scene}.jpg").is_file()


def _build_dense(pos: torch.Tensor, output_size: int) -> dict:
    bbs = GraspRectangles.load_from_grasp_anything_file(
        type("P", (), {"__init__": lambda self, p: None})(),
        scale=output_size / 416.0,
    ) if False else None  # placeholder, replaced below


def dense_maps(stem: str, pos_path: Path, output_size: int) -> dict:
    bbs = GraspRectangles.load_from_grasp_anything_file(pos_path, scale=output_size / 416.0)
    pos_img, ang_img, width_img = bbs.draw((output_size, output_size))
    width_img = np.clip(width_img, 0.0, output_size / 2) / (output_size / 2)
    finite = bool(np.isfinite(pos_img).all() and np.isfinite(ang_img).all() and np.isfinite(width_img).all())
    return {
        "pos_shape": list(pos_img.shape),
        "ang_shape": list(ang_img.shape),
        "width_shape": list(width_img.shape),
        "finite": finite,
        "rectangles": len(bbs.grs),
        "pos_img": pos_img,
        "ang_img": ang_img,
        "width_img": width_img,
    }


def save_qualitative(out_dir: Path, stem: str, rgb_chw: np.ndarray,
                     pos_img: np.ndarray, ang_img: np.ndarray,
                     width_img: np.ndarray) -> None:
    qdir = out_dir / "qualitative"
    qdir.mkdir(parents=True, exist_ok=True)
    rgb = (np.transpose(rgb_chw, (1, 2, 0)) - rgb_chw.min()) / max(1e-6, (rgb_chw.max() - rgb_chw.min()))
    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
    overlay = rgb.copy()
    mask = pos_img > 0.5
    overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([255, 64, 64])).astype(np.uint8)
    PILImage.fromarray(rgb).save(qdir / f"{stem}_rgb.png")
    PILImage.fromarray(overlay).save(qdir / f"{stem}_gt.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stems", required=True)
    parser.add_argument(
        "--instructions-dir",
        default="/mnt/data/grasp-anything-lgd/data/processed/grasp-anything-pp/grasp_instructions/grasp_instructions",
    )
    parser.add_argument(
        "--positive-dir",
        default="/mnt/data/grasp-anything-lgd/data/processed/grasp-anything-pp/grasp_label_positive/grasp_label_positive",
    )
    parser.add_argument("--image-dir", default=None,
                        help="Directory containing <scene>.jpg files; if omitted or a stem's RGB is missing, only dense GT checks run")
    parser.add_argument("--out", default="outputs/batch_smoke")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-stems", type=int, default=0,
                        help="Optional cap on the number of stems processed (0 = all)")
    args = parser.parse_args()

    stems_path = Path(args.stems)
    instr_dir = Path(args.instructions_dir)
    pos_dir = Path(args.positive_dir)
    image_dir = Path(args.image_dir) if args.image_dir else None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qualitative").mkdir(parents=True, exist_ok=True)

    stems = load_stems(stems_path)
    if args.max_stems > 0:
        stems = stems[: args.max_stems]
    if not stems:
        print(f"[FAIL] no stems loaded from {stems_path}", file=sys.stderr)
        return 2

    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda")
    print(f"device: {device}, stems: {len(stems)}, image_dir: {image_dir}")

    dataset = BatchDataset(instr_dir, pos_dir, image_dir, stems, output_size=224)
    net = GenerativeResnet(input_channels=3, dropout=False, channel_size=32).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    metrics: list[dict] = []
    summary_lines: list[str] = []

    for idx, stem in enumerate(stems):
        info: dict = {"stem": stem, "status": "pending"}
        try:
            instruction = dataset._instruction(stem)
        except Exception as exc:
            info["status"] = f"FAIL-instruction:{type(exc).__name__}"
            metrics.append(info)
            summary_lines.append(f"{stem}\t{info['status']}")
            print(f"  [{idx}] {stem} -> {info['status']}")
            continue

        info["instruction_preview"] = (instruction[:80] + "...") if len(instruction) > 80 else instruction

        # dense GT generation always
        pos_path = pos_dir / f"{stem}.pt"
        try:
            dense = dense_maps(stem, pos_path, output_size=224)
        except Exception as exc:
            info["status"] = f"FAIL-dense:{type(exc).__name__}"
            metrics.append(info)
            summary_lines.append(f"{stem}\t{info['status']}")
            print(f"  [{idx}] {stem} -> {info['status']}")
            continue
        info["pos_map_shape"] = dense["pos_shape"]
        info["ang_map_shape"] = dense["ang_shape"]
        info["width_map_shape"] = dense["width_shape"]
        info["gt_rectangles"] = dense["rectangles"]
        info["dense_finite"] = dense["finite"]
        if not dense["finite"]:
            info["status"] = "FAIL-dense-nonfinite"
            metrics.append(info)
            summary_lines.append(f"{stem}\t{info['status']}")
            continue

        if not dataset.has_image(stem):
            info["status"] = "SKIP-RGB"
            info["loss"] = None
            info["correct"] = None
            metrics.append(info)
            summary_lines.append(f"{stem}\tSKIP-RGB\t{dense['rectangles']} gt rectangles")
            print(f"  [{idx}] {stem} -> SKIP-RGB (dense GT only)")
            continue

        # Full RGB chain
        try:
            rgb_chw = dataset._image(stem)
            xb = torch.from_numpy(rgb_chw).unsqueeze(0).to(device)
            yb = (
                torch.from_numpy(np.expand_dims(dense["pos_img"], 0).astype(np.float32)).unsqueeze(0).to(device),
                torch.from_numpy(np.cos(2 * np.expand_dims(dense["ang_img"], 0)).astype(np.float32)).unsqueeze(0).to(device),
                torch.from_numpy(np.sin(2 * np.expand_dims(dense["ang_img"], 0)).astype(np.float32)).unsqueeze(0).to(device),
                torch.from_numpy(np.expand_dims(dense["width_img"], 0).astype(np.float32)).unsqueeze(0).to(device),
            )
            prompt = instruction
            query = instruction

            out = net(xb, [prompt], [query])
            for t in out:
                assert tuple(t.shape) == (1, 1, 224, 224), f"unexpected forward shape {tuple(t.shape)}"
            lossd = net.compute_loss(xb, yb, [prompt], [query])
            optimizer.zero_grad()
            lossd["loss"].backward()
            optimizer.step()
            q_img, ang_img, width_img = post_process_output(
                lossd["pred"]["pos"], lossd["pred"]["cos"],
                lossd["pred"]["sin"], lossd["pred"]["width"],
            )
            gt_bbs = dataset.get_gtbb(idx)
            correct = evaluation.calculate_iou_match(
                q_img, ang_img, gt_bbs, no_grasps=1,
                grasp_width=width_img, threshold=args.threshold,
            )
            info["loss"] = float(lossd["loss"].item())
            info["correct"] = bool(correct)
            info["status"] = "OK"
            try:
                save_qualitative(out_dir, stem, rgb_chw, dense["pos_img"], dense["ang_img"], dense["width_img"])
            except Exception as exc:
                info["qualitative_warning"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            info["status"] = f"FAIL-pipeline:{type(exc).__name__}:{exc}"
        metrics.append(info)
        summary_lines.append(
            f"{stem}\t{info['status']}\tloss={info.get('loss')}\tcorrect={info.get('correct')}"
        )
        print(f"  [{idx}] {stem} -> {info['status']} loss={info.get('loss')} correct={info.get('correct')}")

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"device": str(device), "image_dir": str(image_dir) if image_dir else None,
                   "stems": metrics}, f, indent=2)
    summary_path = out_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        total = len(metrics)
        ok = sum(1 for m in metrics if m["status"] == "OK")
        skip = sum(1 for m in metrics if m["status"] == "SKIP-RGB")
        fail = total - ok - skip
        f.write(f"total: {total}\nok: {ok}\nskip_rgb: {skip}\nfail: {fail}\n")
        f.write("\n".join(summary_lines) + "\n")
    print(f"wrote {metrics_path}")
    print(f"wrote {summary_path}")
    print(f"summary: {ok}/{total} OK, {skip} SKIP-RGB, {fail} FAIL")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
