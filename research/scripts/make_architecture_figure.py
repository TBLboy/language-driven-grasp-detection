"""Render the compact architecture figure used in the CVPR-style paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "research" / "paper"


def box(ax, x, y, w, h, text, fc="#f7f9fc", ec="#334155", fontsize=12, bold=False):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.03",
        linewidth=1.3,
        edgecolor=ec,
        facecolor=fc,
        mutation_aspect=1.0,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color="#0f172a",
    )


def arrow(ax, x1, y1, x2, y2, label=None, color="#64748b"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.4,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )
    if label:
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2 + 0.12,
            label,
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#475569",
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.0, 5.0), dpi=200)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    box(ax, 0.9, 4.2, 1.7, 0.75, "RGB image\n$I$", fontsize=11, bold=True)
    box(ax, 0.9, 2.8, 1.7, 0.75, "Instruction\n$T$", fontsize=11, bold=True)
    box(ax, 2.9, 4.2, 1.7, 0.7, "Vision encoder", fontsize=10.5)
    box(ax, 2.9, 2.8, 1.7, 0.7, "Text encoder", fontsize=10.5)
    box(ax, 5.0, 3.5, 1.9, 1.0, "ALBEF\nvision-language fusion", fontsize=10.5, bold=True)
    box(ax, 7.2, 4.2, 1.8, 0.7, "Visual\nfeature $F_v$", fontsize=10)
    box(ax, 7.2, 2.5, 1.8, 0.7, "Language map\n$y_{\\mathrm{view}}$", fontsize=10)

    arrow(ax, 1.75, 4.2, 2.0, 4.2)
    arrow(ax, 1.75, 2.8, 2.0, 2.8)
    arrow(ax, 3.75, 4.2, 4.55, 3.85)
    arrow(ax, 3.75, 2.8, 4.55, 3.2)
    arrow(ax, 6.0, 3.5, 6.3, 4.2, label="", color="#2563eb")
    arrow(ax, 6.0, 3.5, 6.3, 2.5, label="", color="#2563eb")

    box(ax, 8.65, 3.5, 1.75, 1.4, "LSAR\nfixed scale\n$s=0.01$", fontsize=10.5, bold=True, fc="#eff6ff", ec="#1d4ed8")
    box(ax, 10.45, 3.5, 1.35, 0.9, "GG-CNN\ndecoder", fontsize=10.5, bold=True)
    box(ax, 11.75, 4.4, 1.25, 0.7, "Diffusion\ngeneration", fontsize=9.5, bold=True)
    box(ax, 11.75, 2.6, 1.25, 0.7, "Grasp rectangle\n$(x,y,w,h,\\theta)$", fontsize=9.5, bold=True, fc="#ecfdf5", ec="#047857")

    arrow(ax, 8.0, 4.2, 7.75, 4.2, label="concat")
    arrow(ax, 8.0, 2.5, 7.75, 2.5, label="concat")
    arrow(ax, 9.55, 3.5, 9.85, 3.5, label="residual")
    arrow(ax, 11.1, 3.5, 11.45, 3.5)
    arrow(ax, 12.35, 4.4, 12.35, 3.4, label="dense maps")
    arrow(ax, 12.35, 3.4, 12.35, 3.0, label="decode", color="#047857")

    box(ax, 9.6, 1.55, 2.3, 0.55, "Auxiliary affordance map", fontsize=9, fc="#f8fafc", ec="#94a3b8")
    arrow(ax, 8.65, 2.8, 9.05, 1.9, label="", color="#94a3b8", )
    ax.text(
        5.0,
        1.4,
        "Training: $\\mathcal{L}=\\mathcal{L}_{\\mathrm{diffusion}} + \\mathcal{L}_{\\mathrm{dense}} + 0.05\\,\\mathcal{L}_{\\mathrm{aff}}$",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#0f172a",
    )

    fig.savefig(OUT_DIR / "architecture.pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT_DIR / "architecture.png", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(OUT_DIR / "architecture.pdf")


if __name__ == "__main__":
    main()
