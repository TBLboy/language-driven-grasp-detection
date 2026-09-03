"""Minimal loaders for a small Grasp-Anything++ smoke fixture.

This adapter intentionally keeps the official LGD grasp representation:
it converts positive `[N, 6]` labels into dense `pos/cos/sin/width` maps so
the existing model and evaluation code can be reused unchanged.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from PIL import Image as PILImage
from skimage.transform import resize

from utils.dataset_processing.grasp import GraspRectangles


class GraspAnythingPPSampleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str | Path,
        stems: list[str],
        output_size: int = 224,
    ):
        self.root = Path(root)
        self.stems = stems
        self.output_size = output_size

    def __len__(self) -> int:
        return len(self.stems)

    def _instruction(self, stem: str) -> str:
        path = self.root / "grasp_instructions" / f"{stem}.pkl"
        return pickle.loads(path.read_bytes())

    def _image(self, stem: str) -> np.ndarray:
        scene = stem.rsplit("_", 2)[0]
        path = self.root / "image" / f"{scene}.jpg"
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
        path = self.root / "grasp_label_positive" / f"{stem}.pt"
        bbs = GraspRectangles.load_from_grasp_anything_file(
            path,
            scale=self.output_size / 416.0,
        )
        if rot:
            bbs.rotate(rot, (self.output_size // 2, self.output_size // 2))
        if zoom != 1.0:
            bbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        return bbs

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        rgb = self._image(stem)
        bbs = self.get_gtbb(idx)
        instruction = self._instruction(stem)

        pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        width_img = np.clip(width_img, 0.0, self.output_size / 2) / (
            self.output_size / 2
        )

        def to_tensor(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.expand_dims(arr, 0).astype(np.float32))

        x = torch.from_numpy(rgb.astype(np.float32))
        y = (
            to_tensor(pos_img),
            to_tensor(np.cos(2 * ang_img)),
            to_tensor(np.sin(2 * ang_img)),
            to_tensor(width_img),
        )
        return x, y, idx, 0.0, 1.0, instruction, instruction
