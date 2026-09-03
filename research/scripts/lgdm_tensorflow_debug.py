#!/usr/bin/env python3
"""Debug LGDM conditioning tensors and candidate LSAR insertion points.

The official LGDM code computes an ALBEF fusion output `y` but the
`img = torch.clone(img).detach() + y` line is commented out. In addition, the
`image_atts` returned by ALBEF is an all-ones mask, so the RGB gating path does
not carry language information. This script records the actual tensor shapes
and gradient flow for both behaviors:

- official forward: y exists but has no gradient path into the grasp output
- --inject-y forward: y is added to the GG-CNN conv3 feature, proving the
  shapes match and the text branch can be connected
"""

from __future__ import annotations

import argparse
import json
import os
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

from inference.models.lgdm.network import LGDM  # noqa: E402
from utils.dataset_processing.grasp import GraspRectangles  # noqa: E402
from utils.model_util import create_diffusion  # noqa: E402


def load_stem(tsv: Path, idx: int = 0) -> dict:
    with tsv.open(encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line_no, line in enumerate(f):
            if line_no == idx:
                return dict(zip(header, line.rstrip("\n").split("\t")))
    raise RuntimeError(f"not enough rows in {tsv}")


def load_rgb(path: Path) -> torch.Tensor:
    with PILImage.open(path) as img:
        rgb = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    rgb = resize(rgb, (224, 224), anti_aliasing=True, preserve_range=False)
    rgb = rgb.astype(np.float32)
    rgb -= rgb.mean(axis=(0, 1), keepdims=True)
    return torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0)


def dense_maps(positive_dir: Path, stem: str) -> tuple[torch.Tensor, torch.Tensor]:
    bbs = GraspRectangles.load_from_grasp_anything_file(
        positive_dir / f"{stem}.pt", scale=224 / 416.0
    )
    pos, ang, width = bbs.draw((224, 224))
    width = np.clip(width, 0.0, 112.0) / 112.0
    pos_t = torch.from_numpy(np.expand_dims(pos, 0).astype(np.float32)).unsqueeze(0)
    ang_t = torch.from_numpy(np.expand_dims(ang, 0).astype(np.float32)).unsqueeze(0)
    width_t = torch.from_numpy(np.expand_dims(width, 0).astype(np.float32)).unsqueeze(0)
    return pos_t, ang_t, width_t


def build_debug_forward(inject_y: bool):
    def forward(self, x, img, t, query, alpha, idx, prompt=None):
        device = img.device
        text_input = self.tokenizer(
            query, padding="longest", max_length=30, return_tensors="pt"
        ).to(device)
        image_atts, y = self.albef(img, text_input, alpha, idx)
        full_image_atts = self._process_attention_mask(
            image_atts=image_atts
        ).to(device)
        r_channel, g_channel, b_channel = img[:, 0], img[:, 1], img[:, 2]
        r_channel = r_channel * full_image_atts
        g_channel = g_channel * full_image_atts
        b_channel = b_channel * full_image_atts
        img = torch.cat(
            [
                r_channel.unsqueeze(1),
                g_channel.unsqueeze(1),
                b_channel.unsqueeze(1),
            ],
            dim=1,
        )

        y0 = y[:, 0].unsqueeze(1)
        y_flatten = self.y_flatten(y0)
        y_view = y_flatten.view(-1, 8, 19, 19)
        self._last_y0 = y0
        self._last_y0.retain_grad()

        img = torch.relu(self.conv1(img))
        img = torch.relu(self.conv2(img))
        conv3 = torch.relu(self.conv3(img))
        if inject_y:
            img = conv3 + y_view
            img = torch.relu(img)
        else:
            img = conv3

        img = torch.relu(self.convt1(img))
        img = torch.relu(self.convt2(img))
        img = torch.relu(self.convt3(img))

        pos_output = self.pos_output(img)
        cos_output = self.cos_output(img)
        sin_output = self.sin_output(img)
        width_output = self.width_output(img)
        self.guiding_point = pos_output
        pos_output = x + pos_output
        self.pos_output_str = pos_output
        self.cos_output_str = cos_output
        self.sin_output_str = sin_output
        self.width_output_str = width_output

        self.debug_shapes = {
            "image_atts": list(image_atts.shape),
            "full_image_atts": list(full_image_atts.shape),
            "albef_y": list(y.shape),
            "y0": list(y0.shape),
            "y_flatten": list(y_flatten.shape),
            "y_view_8x19x19": list(y_view.shape),
            "gated_rgb": list(img.shape) if False else list(
                torch.cat(
                    [
                        r_channel.unsqueeze(1),
                        g_channel.unsqueeze(1),
                        b_channel.unsqueeze(1),
                    ],
                    dim=1,
                ).shape
            ),
            "conv3_8x19x19": list(conv3.shape),
            "decoder_output": list(
                torch.relu(self.convt3(torch.relu(self.convt2(torch.relu(self.convt1(conv3)))))).shape
            ),
            "pos_output": list(pos_output.shape),
            "cos_output": list(cos_output.shape),
            "sin_output": list(sin_output.shape),
            "width_output": list(width_output.shape),
        }
        self.debug_values = {
            "full_image_atts_min": float(full_image_atts.min().item()),
            "full_image_atts_max": float(full_image_atts.max().item()),
            "full_image_atts_unique": full_image_atts.unique().cpu().tolist(),
        }
        return pos_output

    return forward


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", default=str(ROOT / "research/smoke-data/train_subset_100.tsv"))
    parser.add_argument("--stem-index", type=int, default=0)
    parser.add_argument(
        "--instruction-dir",
        default="/mnt/data/grasp-anything-lgd/data/processed/grasp-anything-pp/grasp_instructions/grasp_instructions",
    )
    parser.add_argument(
        "--positive-dir",
        default="/mnt/data/grasp-anything-lgd/data/processed/grasp-anything-pp/grasp_label_positive/grasp_label_positive",
    )
    parser.add_argument(
        "--image-dir",
        default="/mnt/data/grasp-anything-lgd/data/processed/grasp-anything/images",
    )
    parser.add_argument("--out", default="outputs/lgdm_tensorflow_debug.json")
    parser.add_argument("--inject-y", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    os.chdir(ROOT / "LGD-main")
    row = load_stem(Path(args.tsv), args.stem_index)
    stem = row["stem"]
    instruction = pickle.loads(
        (Path(args.instruction_dir) / f"{stem}.pkl").read_bytes()
    )
    xb = load_rgb(Path(args.image_dir) / f"{stem.rsplit('_', 2)[0]}.jpg")
    pos_gt, ang_gt, width_gt = dense_maps(Path(args.positive_dir), stem)
    cos_gt = torch.cos(2 * ang_gt)
    sin_gt = torch.sin(2 * ang_gt)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    xb = xb.to(device)
    pos_gt = pos_gt.to(device)
    print(f"stem: {stem}")
    print(f"instruction: {instruction}")
    print(f"device: {device}")

    net = LGDM(input_channels=3).to(device)
    net.forward = build_debug_forward(inject_y=args.inject_y).__get__(
        net, type(net)
    )

    diffusion = create_diffusion()
    t = torch.tensor([500], dtype=torch.long, device=device)
    x_t = diffusion.q_sample(pos_gt, t)
    idx = torch.zeros(1, dtype=torch.long, device=device)
    net.train()

    losses = diffusion.training_losses(
        net,
        pos_gt,
        xb,
        t,
        [instruction],
        0.4,
        idx,
    )
    loss = (losses["loss"] * torch.ones(1, device=device)).mean()
    loss = (
        loss
        + torch.nn.functional.mse_loss(net.cos_output_str, cos_gt.to(device))
        + torch.nn.functional.mse_loss(net.sin_output_str, sin_gt.to(device))
        + torch.nn.functional.mse_loss(net.width_output_str, width_gt.to(device))
    )

    y0 = net._last_y0
    y_grad = None
    full_att_grad = "not-computed"

    loss.backward()
    if y0 is not None and y0.grad is not None:
        y_grad = {
            "shape": list(y0.grad.shape),
            "finite": bool(torch.isfinite(y0.grad).all().item()),
        }
    else:
        y_grad = None

    albef_text = net.albef.text_encoder
    text_param = next(p for p in albef_text.parameters() if p.requires_grad)
    text_grad = (
        {
            "name": None,
            "finite": bool(torch.isfinite(text_param.grad).all().item()),
        }
        if text_param.grad is not None
        else None
    )

    report = {
        "stem": stem,
        "instruction": instruction,
        "inject_y": bool(args.inject_y),
        "shapes": net.debug_shapes,
        "full_image_atts_values": net.debug_values,
        "loss_terms": {
            "diffusion_mse": float(losses["mse"].mean().item()),
            "diffusion_contr": float(losses["contr"].mean().item()),
            "diffusion_loss": float(loss.item()),
        },
        "gradient": {
            "y_flatten_input_grad": y_grad,
            "albef_text_param_grad": text_grad,
        },
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
