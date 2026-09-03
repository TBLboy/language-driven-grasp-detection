#!/usr/bin/env bash
# prepare_rgb_archive.sh
#
# Inspect /mnt/data/.../raw/grasp-anything/image_part_{aa,ab} and decide
# whether they are split parts of the same archive. Default mode is
# "identify-only": it prints stat / file / magic header / first bytes and
# exits. No merging happens unless `--auto-merge` is passed AND every
# evidence check agrees.
#
# Usage:
#   prepare_rgb_archive.sh                     # identify only (default)
#   prepare_rgb_archive.sh --auto-merge        # merge when evidence supports
#   prepare_rgb_archive.sh --auto-merge --keep-raw   # keep aa/ab after merge
#
# Expected sizes (bytes):
#   image_part_aa = 34359738368
#   image_part_ab = 30653099134
set -euo pipefail

RAW_DIR="${RAW_DIR:-/mnt/data/grasp-anything-lgd/data/raw/grasp-anything}"
OUT_DIR="${OUT_DIR:-/mnt/data/grasp-anything-lgd/data/processed/grasp-anything/images}"
AA="$RAW_DIR/image_part_aa"
AB="$RAW_DIR/image_part_ab"
EXPECTED_AA=34359738368
EXPECTED_AB=30653099134
MERGED="${MERGED:-$RAW_DIR/image_archive}"

MODE="identify"
KEEP_RAW=0
for arg in "$@"; do
  case "$arg" in
    --auto-merge) MODE="merge" ;;
    --keep-raw)   KEEP_RAW=1 ;;
    -h|--help)
      sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

log()  { printf '[prepare_rgb] %s\n' "$*"; }
fail() { printf '[prepare_rgb][FAIL] %s\n' "$*" >&2; exit 1; }

log "AA = $AA"
log "AB = $AB"
log "OUT_DIR = $OUT_DIR"
log "MERGED (only when --auto-merge) = $MERGED"

if [[ ! -f "$AA" ]]; then fail "AA part missing"; fi
if [[ ! -f "$AB" ]]; then fail "AB part missing"; fi

aa_size=$(stat -c '%s' "$AA")
ab_size=$(stat -c '%s' "$AB")
log "AA size = $aa_size (expected $EXPECTED_AA)"
log "AB size = $ab_size (expected $EXPECTED_AB)"

aa_size_ok=0; [[ "$aa_size" -eq "$EXPECTED_AA" ]] && aa_size_ok=1
ab_size_ok=0; [[ "$ab_size" -eq "$EXPECTED_AB" ]] && ab_size_ok=1
if [[ "$aa_size_ok" -ne 1 ]]; then log "AA size mismatch (got $aa_size, expected $EXPECTED_AA)"; fi
if [[ "$ab_size_ok" -ne 1 ]]; then log "AB size mismatch (got $ab_size, expected $EXPECTED_AB)"; fi

aa_type=$(file -b "$AA")
ab_type=$(file -b "$AB")
log "AA file type: $aa_type"
log "AB file type: $ab_type"

log "AA first 32 bytes:"
xxd -l 32 "$AA"
log "AB first 32 bytes:"
xxd -l 32 "$AB"

aa_magic=$(head -c 4 "$AA" | od -An -tx1 | tr -d ' \n')
ab_magic=$(head -c 4 "$AB" | od -An -tx1 | tr -d ' \n')
log "AA magic = $aa_magic"
log "AB magic = $ab_magic"

# Heuristic: split archives produced by `split -b` typically have no
# recognizable header on the first part if the split point fell inside a
# compressed stream. The reliable evidence is the exact byte sizes matching
# the expected sizes recorded in DEC-001.
evidence_ok=0
if [[ "$aa_size_ok" -eq 1 && "$ab_size_ok" -eq 1 ]]; then
  evidence_ok=1
  log "evidence_ok=1 (sizes match expected, no magic header required)"
else
  log "evidence_ok=0: sizes do not match expected; refuse to merge automatically"
fi

if [[ "$MODE" == "identify" ]]; then
  log "identify-only mode; pass --auto-merge to attempt merging"
  exit 0
fi

if [[ "$evidence_ok" -ne 1 ]]; then
  fail "auto-merge refused; sizes do not match DEC-001 record"
fi

if [[ -e "$MERGED" ]]; then
  log "merged archive already exists at $MERGED; skipping cat"
else
  log "cat $AA $AB > $MERGED"
  cat "$AA" "$AB" > "$MERGED"
fi

log "MERGED file type:"
file -b "$MERGED"
log "MERGED first 32 bytes:"
xxd -l 32 "$MERGED"

log "MERGED listing (first 8 entries):"
unzip -l "$MERGED" | head -n 8 || log "unzip -l failed; archive format not recognised"

if [[ "$KEEP_RAW" -ne 1 ]]; then
  log "--keep-raw not set; aa/ab remain in $RAW_DIR"
fi
log "done"
