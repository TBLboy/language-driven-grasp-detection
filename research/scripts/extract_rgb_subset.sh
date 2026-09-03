#!/usr/bin/env bash
# Extract only the <scene>.jpg files needed by a cross-scene stem TSV.
#
# The RGB upload is stored as two raw split parts (image_part_aa and
# image_part_ab). Merging them produces the .zip expected by the original
# dataset. This script merges only when both sizes match, extracts the listed
# members, and deletes the temporary merged archive by default.
#
# Usage:
#   extract_rgb_subset.sh --stems research/smoke-data/train_subset_100.tsv
#   extract_rgb_subset.sh --stems file.txt --keep-archive
set -euo pipefail

RAW_DIR="${RAW_DIR:-/mnt/data/grasp-anything-lgd/data/raw/grasp-anything}"
OUT_DIR="${OUT_DIR:-/mnt/data/grasp-anything-lgd/data/processed/grasp-anything/images}"
AA="$RAW_DIR/image_part_aa"
AB="$RAW_DIR/image_part_ab"
EXPECTED_AA=34359738368
EXPECTED_AB=30653099134
MERGED="${MERGED:-$RAW_DIR/image_archive}"

STEMS_FILE=""
KEEP_ARCHIVE=0
while (($# > 0)); do
  case "$1" in
    --stems) STEMS_FILE="$2"; shift 2 ;;
    --keep-archive) KEEP_ARCHIVE=1; shift ;;
    -h|--help) sed -n '1,24p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$STEMS_FILE" ]]; then
  echo "error: --stems is required (TSV with scene column or one stem per line)" >&2
  exit 2
fi
if [[ ! -f "$STEMS_FILE" ]]; then
  echo "error: stems file not found: $STEMS_FILE" >&2
  exit 2
fi
if [[ ! -f "$AA" || ! -f "$AB" ]]; then
  echo "error: RGB raw parts are missing under $RAW_DIR" >&2
  exit 2
fi

aa_size=$(stat -c '%s' "$AA")
ab_size=$(stat -c '%s' "$AB")
if [[ "$aa_size" -ne "$EXPECTED_AA" || "$ab_size" -ne "$EXPECTED_AB" ]]; then
  echo "error: raw part sizes do not match ($aa_size / $ab_size)" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

# TSV files from prepare_training_subset.py use a scene column; plain text
# stem files are also accepted and parsed as <scene>_<obj>_<part>.
if awk -F '\t' 'NR == 1 { exit }' "$STEMS_FILE" 2>/dev/null; then
  scenes=$(awk -F '\t' 'NR > 1 && $2 != "" { print $2 }' "$STEMS_FILE" | sort -u)
else
  scenes=$(awk '{ gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0); if ($0 != "" && $0 !~ /^#/) { sub(/_[0-9]+_[0-9]+$/, "", $0); print } }' "$STEMS_FILE" | sort -u)
fi

if [[ -z "$scenes" ]]; then
  echo "error: no scenes parsed from $STEMS_FILE" >&2
  exit 2
fi

if [[ ! -f "$MERGED" ]]; then
  echo "[extract_rgb] merging aa+ab into $MERGED"
  cat "$AA" "$AB" > "$MERGED"
fi

echo "[extract_rgb] extracting $(wc -l <<<"$scenes") unique scene images"
files=""
while IFS= read -r scene; do
  files="$files image/$scene.jpg"
done <<<"$scenes"

# -o overwrite, -j junk paths so output is directly $OUT_DIR/<scene>.jpg
unzip -o -j "$MERGED" $files -d "$OUT_DIR"

extracted=0
missing=0
while IFS= read -r scene; do
  if [[ -f "$OUT_DIR/$scene.jpg" ]]; then
    extracted=$((extracted + 1))
  else
    echo "missing scene: $scene"
    missing=$((missing + 1))
  fi
done <<<"$scenes"

echo "[extract_rgb] extracted=$extracted missing=$missing total=$(wc -l <<<"$scenes")"

if [[ "$KEEP_ARCHIVE" -ne 1 ]]; then
  echo "[extract_rgb] deleting temporary merged archive"
  rm -f "$MERGED"
fi

if [[ "$extracted" -eq "$(wc -l <<<"$scenes")" ]]; then
  exit 0
fi
exit 1
