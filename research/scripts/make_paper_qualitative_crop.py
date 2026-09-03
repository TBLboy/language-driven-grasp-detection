"""Create a compact landscape crop of the paired qualitative paper figure."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE = (
    ROOT
    / "outputs"
    / "lgdm_10k"
    / "paired_visuals_final_005"
    / "paired_qualitative.png"
)
TARGET = ROOT / "research" / "paper" / "qualitative_paper.png"

# The original figure is a 6-row x 4-column grid saved at 2400x3600.
# Select part-level cases that show LSAR correcting the baseline where useful:
# row 1: highlighter, row 4: apple stem, row 5: keychain keys.
SELECTED_ROW_INDICES = [1, 4, 5]
ROW_HEIGHT = 600


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"source figure not found: {SOURCE}")
    with Image.open(SOURCE) as img:
        width, height = img.size
        assert height == 3600 and width == 2400, (width, height)
        rows = [img.crop((0, i * ROW_HEIGHT, width, (i + 1) * ROW_HEIGHT)) for i in SELECTED_ROW_INDICES]
    combined = Image.new("RGB", (width, ROW_HEIGHT * len(rows)), "white")
    for y, row in enumerate(rows):
        combined.paste(row, (0, y * ROW_HEIGHT))
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    combined.save(TARGET, optimize=True)
    print(TARGET, combined.size)


if __name__ == "__main__":
    main()
