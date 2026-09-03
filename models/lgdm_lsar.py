"""LGDM conditioning variants used for the LSAR baseline comparison.

The official `LGDM` model computes an ALBEF fusion output `y` but does not
connect it to the GG-CNN decoder (`img = torch.clone(img).detach() + y` is
commented out in the upstream source). This module keeps the official model
untouched and provides a subclass with three reproducible conditioning modes:

- `none`: official LGDM forward, no y injection
- `plain-y`: inject the raw ALBEF text feature at the GG-CNN conv3 level
- `lsar`: refine the text feature with our Language-conditioned Spatial
  Affordance Refinement module before injecting it at conv3

Only the conditioning branch changes; grasp representation, diffusion
schedule, dense-map decoding, and evaluation remain identical.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "LGD-main") not in sys.path:
    sys.path.insert(0, str(ROOT / "LGD-main"))

from inference.models.lgdm.network import LGDM  # noqa: E402


class SpatialAffordanceRefinement(nn.Module):
    """Lightweight language-conditioned spatial affordance refinement.

    Inputs are the language-conditioned `y_view` (8x19x19) and the visual
    GG-CNN conv3 feature with the same shape. The module computes a residual
    correction that can be added to conv3 before the decoder.
    """

    def __init__(
        self,
        channels: int = 8,
        hidden: int = 16,
        init_scale: float = 0.1,
        trainable_scale: bool = True,
    ) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(hidden, channels, kernel_size=1)
        self.aff_head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        # Start with a small non-zero residual. A zero scale would also zero
        # the gradient through the residual branch and the module never learns.
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        if not trainable_scale:
            self.scale.requires_grad_(False)

    def forward(
        self,
        visual: torch.Tensor,
        text_map: torch.Tensor,
        return_affordance: bool = False,
    ):
        hidden = self.fuse(torch.cat([visual, text_map], dim=1))
        residual = self.scale * self.proj(hidden)
        if return_affordance:
            return residual, self.aff_head(hidden)
        return residual


class LGDMWithConditioning(LGDM):
    def __init__(
        self,
        condition_mode: str = "none",
        input_channels: int = 3,
        output_channels: int = 1,
        channel_size: int = 32,
        dropout: bool = False,
        prob: float = 0.0,
        clip_version: str = "ViT-B/32",
        lsar_scale: float = 0.1,
        lsar_scale_trainable: bool = True,
    ) -> None:
        if condition_mode not in {"none", "plain-y", "lsar"}:
            raise ValueError(f"unknown condition_mode: {condition_mode}")
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            channel_size=channel_size,
            dropout=dropout,
            prob=prob,
            clip_version=clip_version,
        )
        self.condition_mode = condition_mode
        if condition_mode == "lsar":
            self.lsar = SpatialAffordanceRefinement(
                init_scale=lsar_scale,
                trainable_scale=lsar_scale_trainable,
            )

    def forward(self, x, img, t, query, alpha, idx, prompt=None):
        device = img.device
        text_input = self.tokenizer(
            query, padding="longest", max_length=30, return_tensors="pt"
        ).to(device)
        image_atts, y = self.albef(img, text_input, alpha, idx)
        self.full_image_atts = self._process_attention_mask(
            image_atts=image_atts
        ).to(device)
        r_channel, g_channel, b_channel = img[:, 0], img[:, 1], img[:, 2]
        r_channel = r_channel * self.full_image_atts
        g_channel = g_channel * self.full_image_atts
        b_channel = b_channel * self.full_image_atts
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
        self.condition_feature = y_view

        img = F.relu(self.conv1(img))
        img = F.relu(self.conv2(img))
        img = F.relu(self.conv3(img))

        if self.condition_mode == "plain-y":
            img = F.relu(img + y_view)
        elif self.condition_mode == "lsar":
            residual, affordance_map = self.lsar(
                img, y_view, return_affordance=True
            )
            self.affordance_map = affordance_map
            img = F.relu(img + residual)

        img = F.relu(self.convt1(img))
        img = F.relu(self.convt2(img))
        img = F.relu(self.convt3(img))

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
        return pos_output
