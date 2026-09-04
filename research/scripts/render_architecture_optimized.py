"""Render a publication-ready architecture figure for a CVPR-style paper.

Design goals
------------
- Preserve the architecture and terminology of the original diagram.
- Keep a restrained paper-friendly palette with one method accent and one output accent.
- Make the main path visually dominant while keeping the affordance branch secondary.
- Place edge labels in whitespace instead of on top of arrows.
- Export vector PDF plus high-resolution PNG with deterministic typography.

Usage
-----
python render_architecture_optimized.py
python render_architecture_optimized.py --out-dir ./figures --stem architecture --dpi 300
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# ---------------------------------------------------------------------------
# Typography / theme
# ---------------------------------------------------------------------------

def choose_serif_font() -> str:
    """Prefer Times-compatible fonts commonly available on paper servers."""
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in ("Times New Roman", "Tinos", "Liberation Serif", "STIX Two Text"):
        if name in installed:
            return name
    return "DejaVu Serif"


FONT = choose_serif_font()

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [FONT],
        "font.size": 9.2,
        "axes.unicode_minus": False,
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT,
        "mathtext.it": FONT,
        "mathtext.bf": FONT,
        "mathtext.default": "regular",
        # Keep text editable/searchable in vector outputs.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


@dataclass(frozen=True)
class Palette:
    text: str = "#172033"
    line: str = "#65748a"
    line_light: str = "#98a6b9"
    border: str = "#334155"
    box: str = "#fbfcfe"
    method: str = "#2563eb"
    method_fill: str = "#f3f7ff"
    output: str = "#047857"
    output_fill: str = "#f1faf6"
    auxiliary_fill: str = "#fafbfc"


C = Palette()


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Node:
    x: float
    y: float
    w: float
    h: float

    @property
    def left(self) -> float:
        return self.x - self.w / 2

    @property
    def right(self) -> float:
        return self.x + self.w / 2

    @property
    def top(self) -> float:
        return self.y + self.h / 2

    @property
    def bottom(self) -> float:
        return self.y - self.h / 2


Side = Literal["left", "right", "top", "bottom"]


def anchor(node: Node, side: Side, offset: float = 0.0) -> tuple[float, float]:
    """Return an anchor point on one side of a node."""
    if side == "left":
        return node.left, node.y + offset
    if side == "right":
        return node.right, node.y + offset
    if side == "top":
        return node.x + offset, node.top
    if side == "bottom":
        return node.x + offset, node.bottom
    raise ValueError(f"Unknown side: {side}")


def midpoint(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    dx: float = 0.0,
    dy: float = 0.0,
) -> tuple[float, float]:
    return ((start[0] + end[0]) / 2 + dx, (start[1] + end[1]) / 2 + dy)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_box(
    ax: Axes,
    node: Node,
    text: str,
    *,
    fontsize: float = 9.2,
    bold: bool = False,
    fc: str = C.box,
    ec: str = C.border,
    lw: float = 0.95,
    linestyle: str = "-",
) -> None:
    patch = FancyBboxPatch(
        (node.left, node.bottom),
        node.w,
        node.h,
        boxstyle="round,pad=0.016,rounding_size=0.030",
        linewidth=lw,
        linestyle=linestyle,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        node.x,
        node.y,
        text,
        ha="center",
        va="center",
        multialignment="center",
        fontsize=fontsize,
        fontweight="semibold" if bold else "normal",
        color=C.text,
        linespacing=1.02,
        zorder=3,
    )


def draw_arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = C.line,
    lw: float = 0.95,
    rad: float = 0.0,
    mutation_scale: float = 10.5,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=lw,
            color=color,
            shrinkA=3.0,
            shrinkB=3.0,
            connectionstyle=f"arc3,rad={rad}",
            capstyle="round",
            joinstyle="round",
            zorder=1,
        )
    )


def edge_label(
    ax: Axes,
    x: float,
    y: float,
    text: str,
    *,
    color: str = C.line,
    fontsize: float = 7.6,
    ha: str = "center",
    va: str = "center",
) -> None:
    # Intentionally no bbox: labels sit in whitespace and do not mask arrows.
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=fontsize,
        color=color,
        zorder=4,
        clip_on=False,
    )


def draw_figure() -> tuple[Figure, Axes]:
    """Build the architecture diagram and return the Matplotlib figure/axes."""
    fig, ax = plt.subplots(figsize=(12.6, 3.95), dpi=220)
    ax.set_xlim(0.0, 14.3)
    ax.set_ylim(0.35, 4.45)
    ax.set_aspect("auto")
    ax.axis("off")

    # ----------------------------- layout ---------------------------------
    # Input / encoder columns
    rgb = Node(0.90, 3.52, 1.48, 0.68)
    instruction = Node(0.90, 2.18, 1.48, 0.68)
    vision_encoder = Node(2.70, 3.52, 1.52, 0.62)
    text_encoder = Node(2.70, 2.18, 1.52, 0.62)

    # Fusion / features
    fusion = Node(4.78, 2.85, 1.82, 1.04)
    visual_feature = Node(6.93, 3.52, 1.78, 0.66)
    language_map = Node(6.93, 2.18, 1.78, 0.66)

    # Main method / decoding / output
    lsar = Node(9.18, 2.85, 1.72, 1.22)
    decoder = Node(11.35, 2.85, 1.48, 0.80)
    diffusion = Node(13.29, 3.48, 1.50, 0.70)
    grasp = Node(13.29, 2.00, 1.72, 0.76)

    # Auxiliary branch and training objective
    affordance = Node(9.18, 0.76, 2.05, 0.48)

    # ----------------------------- boxes ----------------------------------
    draw_box(ax, rgb, "RGB image\n$\\mathrm{I}$", fontsize=9.5, bold=True)
    draw_box(ax, instruction, "Instruction\n$\\mathrm{T}$", fontsize=9.5, bold=True)

    draw_box(ax, vision_encoder, "Vision encoder")
    draw_box(ax, text_encoder, "Text encoder")

    draw_box(
        ax,
        fusion,
        "ALBEF\nvision–language\nfusion",
        fontsize=9.0,
        bold=True,
    )

    draw_box(ax, visual_feature, "Visual feature\n$\\mathrm{F}_{\\mathrm{v}}$", fontsize=9.0)
    draw_box(ax, language_map, "Language map\n$\\mathrm{y}_{\\mathrm{view}}$", fontsize=9.0)

    draw_box(
        ax,
        lsar,
        "LSAR\nfixed scale\n$\\mathrm{s}=0.01$",
        fontsize=9.2,
        bold=True,
        fc=C.method_fill,
        ec=C.method,
        lw=1.15,
    )

    draw_box(ax, decoder, "GG-CNN\ndecoder", fontsize=9.2, bold=True)
    draw_box(ax, diffusion, "Diffusion\ngeneration", fontsize=8.9, bold=True)
    draw_box(
        ax,
        grasp,
        "Grasp rectangle\n$(\\mathrm{x},\\mathrm{y},\\mathrm{w},\\mathrm{h},\\mathrm{\\theta})$",
        fontsize=8.6,
        bold=True,
        fc=C.output_fill,
        ec=C.output,
        lw=1.10,
    )
    draw_box(
        ax,
        affordance,
        "Auxiliary affordance map",
        fontsize=8.4,
        fc=C.auxiliary_fill,
        ec=C.line_light,
        lw=0.85,
        linestyle="--",
    )

    # ----------------------------- arrows ---------------------------------
    draw_arrow(ax, anchor(rgb, "right"), anchor(vision_encoder, "left"))
    draw_arrow(ax, anchor(instruction, "right"), anchor(text_encoder, "left"))

    draw_arrow(ax, anchor(vision_encoder, "right"), anchor(fusion, "left", +0.25))
    draw_arrow(ax, anchor(text_encoder, "right"), anchor(fusion, "left", -0.25))

    # Fusion outputs get a subtle method-colored accent.
    draw_arrow(
        ax,
        anchor(fusion, "right", +0.25),
        anchor(visual_feature, "left"),
        color=C.method,
        lw=1.05,
    )
    draw_arrow(
        ax,
        anchor(fusion, "right", -0.25),
        anchor(language_map, "left"),
        color=C.method,
        lw=1.05,
    )

    vf_to_lsar = (anchor(visual_feature, "right"), anchor(lsar, "left", +0.28))
    lm_to_lsar = (anchor(language_map, "right"), anchor(lsar, "left", -0.28))
    draw_arrow(ax, *vf_to_lsar)
    draw_arrow(ax, *lm_to_lsar)

    # One operation label is clearer than two repeated "concat" labels.
    edge_label(ax, 8.12, 2.85, "concat", fontsize=7.7)

    lsar_to_decoder = (anchor(lsar, "right"), anchor(decoder, "left"))
    draw_arrow(ax, *lsar_to_decoder)

    # Keep the residual label centered in a deliberately widened inter-module gap.
    # This reads more naturally than letting the word touch either box.
    rx, ry = midpoint(*lsar_to_decoder, dy=0.17)
    edge_label(ax, rx, ry, "residual", fontsize=7.4)

    draw_arrow(
        ax,
        anchor(decoder, "right", +0.10),
        anchor(diffusion, "left", -0.05),
    )

    # Output path: method result gets the second, semantically meaningful accent.
    diff_to_grasp = (anchor(diffusion, "bottom"), anchor(grasp, "top"))
    draw_arrow(ax, *diff_to_grasp, color=C.output, lw=1.05)
    gx, gy = midpoint(*diff_to_grasp)
    edge_label(ax, gx + 0.24, gy + 0.14, "dense maps", color=C.output, fontsize=7.7, ha="left")
    edge_label(ax, gx + 0.24, gy - 0.10, "decode", color=C.output, fontsize=7.7, ha="left")

    # Auxiliary supervision is deliberately lighter and dashed-boxed.
    draw_arrow(
        ax,
        anchor(lsar, "bottom"),
        anchor(affordance, "top"),
        color=C.line_light,
        lw=0.82,
        mutation_scale=9.5,
    )

    # --------------------------- objective --------------------------------
    ax.text(
        4.65,
        0.74,
        (
            "Training objective:  "
            "$\\mathrm{L}=\\mathrm{L}_{\\mathrm{diffusion}}"
            "+\\mathrm{L}_{\\mathrm{dense}}"
            "+0.05\\,\\mathrm{L}_{\\mathrm{aff}}$"
        ),
        ha="center",
        va="center",
        fontsize=8.8,
        color=C.text,
        zorder=3,
    )

    fig.subplots_adjust(left=0.014, right=0.990, top=0.965, bottom=0.075)
    return fig, ax


def save_figure(fig: Figure, out_dir: Path, stem: str, dpi: int) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"

    common = dict(bbox_inches="tight", pad_inches=0.035, facecolor="white")
    fig.savefig(pdf_path, **common)
    fig.savefig(png_path, dpi=dpi, **common)
    return pdf_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Output directory (default: directory containing this script).",
    )
    parser.add_argument("--stem", default="architecture_optimized", help="Output filename stem.")
    parser.add_argument("--dpi", type=int, default=320, help="PNG export DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fig, _ = draw_figure()
    pdf_path, png_path = save_figure(fig, args.out_dir, args.stem, args.dpi)
    plt.close(fig)

    print(f"font={FONT}")
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
