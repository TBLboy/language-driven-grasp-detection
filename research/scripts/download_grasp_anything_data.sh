#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
DEFAULT_DATA_ROOT="${PROJECT_ROOT}/data"

HF_MIRROR_BASE="${HF_MIRROR_BASE:-https://hf-mirror.com}"
MIN_FREE_BYTES=128849018880    # 120 GiB
WARN_FREE_BYTES=161061273600   # 150 GiB

PP_REPO="airvlab/Grasp-Anything-pp"
BASE_REPO="airvlab/Grasp-Anything"

FILES=(
  "${PP_REPO}|grasp_instructions.zip|1544210262|application/zip"
  "${PP_REPO}|grasp_label_positive.zip|3949367278|application/zip"
  "${BASE_REPO}|image_part_aa|34359738368|application/octet-stream"
  "${BASE_REPO}|image_part_ab|30653099134|application/octet-stream"
)

MODE="download"
ROOT=""
RANGE_PROBE=1
CURL_AUTH_CONFIG=""

cleanup() {
  if [[ -n "$CURL_AUTH_CONFIG" && -f "$CURL_AUTH_CONFIG" ]]; then
    rm -f "$CURL_AUTH_CONFIG"
  fi
}
trap cleanup EXIT

usage() {
  cat <<'EOF'
Usage:
  download_grasp_anything_data.sh [--check] [--root DIR] [--no-range-probe]

Modes:
  --check             Verify token, mirror URLs, sizes, and resume support
  --download          Download raw files with resume (default)

Options:
  --root DIR          Data root; defaults to <project>/data
  --no-range-probe    Skip the 128-byte range probe in --check
  -h, --help          Show this help

The token is read from HF_TOKEN or ~/.cache/huggingface/token.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

info() {
  printf '%s\n' "$*"
}

human_bytes() {
  awk -v n="$1" 'BEGIN {
    split("B KiB MiB GiB TiB", u, " ");
    v = n;
    i = 1;
    while (v >= 1024 && i < 5) {
      v /= 1024;
      i++;
    }
    printf "%.1f %s\n", v, u[i];
  }'
}

check_commands() {
  for cmd in curl stat df mktemp chmod sed awk dirname date; do
    command -v "$cmd" >/dev/null 2>&1 || die "required command not found: $cmd"
  done
}

resolve_token() {
  TOKEN="${HF_TOKEN:-}"
  if [[ -z "$TOKEN" && -f "$HOME/.cache/huggingface/token" ]]; then
    TOKEN="$(tr -d '[:space:]' < "$HOME/.cache/huggingface/token")"
  fi
  if [[ -z "$TOKEN" ]]; then
    die "HF token not found; set HF_TOKEN or log in to ~/.cache/huggingface/token"
  fi
}

setup_curl_auth() {
  if [[ -n "$CURL_AUTH_CONFIG" ]]; then
    return 0
  fi
  CURL_AUTH_CONFIG="$(mktemp "${TMPDIR:-/tmp}/ga-curl-config.XXXXXX")"
  chmod 600 "$CURL_AUTH_CONFIG"
  printf 'header = "Authorization: Bearer %s"\n' "$TOKEN" > "$CURL_AUTH_CONFIG"
}

curl_with_auth() {
  curl --config "$CURL_AUTH_CONFIG" "$@"
}

data_url() {
  local repo="$1"
  local file="$2"
  printf '%s/datasets/%s/resolve/main/%s?download=true' \
    "$HF_MIRROR_BASE" "$repo" "$file"
}

free_bytes_for() {
  local dir="$1"
  while [[ ! -d "$dir" ]]; do
    local parent
    parent="$(dirname "$dir")"
    [[ "$parent" == "$dir" ]] && break
    dir="$parent"
  done
  df -B1 "$dir" | awk 'NR == 2 {print $4}'
}

check_disk() {
  local free_bytes
  free_bytes="$(free_bytes_for "$ROOT")"
  info "[DISK] root=$(human_hard_path "$ROOT") free=$(human_bytes "$free_bytes")"
  if (( free_bytes < MIN_FREE_BYTES )); then
    die "insufficient free disk space for the dataset"
  fi
  if (( free_bytes < WARN_FREE_BYTES )); then
    info "[DISK] warning: free space is below 150 GiB; monitor extraction space too"
  fi
}

human_hard_path() {
  printf '%s' "$1"
}

check_remote_file() {
  local repo="$1"
  local file="$2"
  local expected="$3"
  local expected_ctype="$4"
  local url
  local headers
  local meta
  local http_code content_length accept_ranges content_type
  local ok="1"
  local reason=""

  url="$(data_url "$repo" "$file")"
  info "[CHECK] $file"

  headers="$(curl_with_auth -sSIL --max-time 60 "$url")"
  meta="$(printf '%s\n' "$headers" | sed 's/\r$//' | awk '
    /^HTTP\// { code = $2 }
    tolower($1) == "content-length:" { len = $2 }
    tolower($1) == "accept-ranges:" { ranges = $2 }
    tolower($1) == "content-type:" { ctype = $2 }
    END { printf "%s|%s|%s|%s", code, len, ranges, ctype }
  ')" || die "failed to parse HTTP headers for $file"

  IFS='|' read -r http_code content_length accept_ranges content_type <<< "$meta"

  if [[ "$http_code" != "200" ]]; then
    ok=0
    reason="$reason http=$http_code"
  fi
  if [[ "$content_length" != "$expected" ]]; then
    ok=0
    reason="$reason size=$content_length expected=$expected"
  fi
  if [[ "$accept_ranges" != "bytes" ]]; then
    ok=0
    reason="$reason accept_ranges=$accept_ranges"
  fi
  case "$content_type" in
    "$expected_ctype"|"$expected_ctype;"*)
      ;;
    *)
      ok=0
      reason="$reason content_type=$content_type"
      ;;
  esac

  if [[ "$ok" == "1" && "$RANGE_PROBE" == "1" ]]; then
    local probe
    local probe_code
    local probe_size
    probe="$(mktemp "${TMPDIR:-/tmp}/ga-range-probe.XXXXXX")"
    if ! probe_code="$(
      curl_with_auth -fsSL --max-filesize 4096 --range 0-127 --max-time 60 \
        -o "$probe" \
        -w '%{http_code}' \
        "$url"
    )"; then
      rm -f "$probe"
      die "range probe failed for $file"
    fi
    probe_size="$(stat -c%s "$probe")"
    rm -f "$probe"

    if [[ "$probe_code" != "206" || "$probe_size" != "128" ]]; then
      ok=0
      reason="$reason range=http=$probe_code bytes=$probe_size"
    fi
  fi

  if [[ "$ok" == "1" ]]; then
    if [[ "$RANGE_PROBE" == "1" ]]; then
      info "[OK] $file size=$content_length bytes; resume supported (range=206)"
    else
      info "[OK] $file size=$content_length bytes; range probe disabled"
    fi
  else
    info "[FAIL] $file$reason"
    return 1
  fi
}

check_all_remote() {
  local entry repo file expected ctype
  for entry in "${FILES[@]}"; do
    IFS='|' read -r repo file expected ctype <<< "$entry"
    check_remote_file "$repo" "$file" "$expected" "$ctype"
  done
}

record_manifest() {
  local file="$1"
  local expected="$2"
  local actual="$3"
  local status="$4"
  local manifest="$ROOT/raw/download-manifest.tsv"

  mkdir -p "$ROOT/raw"
  if [[ ! -f "$manifest" ]]; then
    printf 'timestamp\tfile\texpected_bytes\tactual_bytes\tstatus\n' > "$manifest"
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$file" \
    "$expected" \
    "$actual" \
    "$status" >> "$manifest"
}

download_file() {
  local repo="$1"
  local file="$2"
  local expected="$3"
  local ctype="$4"
  local url
  local dest
  local actual
  local raw_dir

  if [[ "$repo" == "$PP_REPO" ]]; then
    raw_dir="$ROOT/raw/grasp-anything-pp"
  else
    raw_dir="$ROOT/raw/grasp-anything"
  fi
  dest="$raw_dir/$file"
  url="$(data_url "$repo" "$file")"

  mkdir -p "$raw_dir"
  if [[ -f "$dest" ]]; then
    actual="$(stat -c%s "$dest")"
    if [[ "$actual" == "$expected" ]]; then
      info "[SKIP] $file already complete"
      record_manifest "$file" "$expected" "$actual" "already-complete"
      return 0
    fi
    if (( actual > expected )); then
      record_manifest "$file" "$expected" "$actual" "too-large"
      die "$file is larger than expected; remove it before retrying"
    fi
    info "[RESUME] $file already has $(human_bytes "$actual"); continuing"
  fi

  info "[DOWNLOAD] $file -> $dest"
  if ! curl_with_auth --fail --location --continue-at - \
    --retry 5 --retry-delay 5 --retry-all-errors \
    --connect-timeout 30 \
    --output "$dest" \
    "$url"; then
    if [[ -f "$dest" ]]; then
      actual="$(stat -c%s "$dest")"
      record_manifest "$file" "$expected" "$actual" "download-interrupted"
    fi
    die "download failed for $file (partial file kept for resume)"
  fi

  actual="$(stat -c%s "$dest")"
  if [[ "$actual" != "$expected" ]]; then
    record_manifest "$file" "$expected" "$actual" "size-mismatch"
    die "size mismatch for $file: expected $expected, got $actual"
  fi

  record_manifest "$file" "$expected" "$actual" "ok"
  info "[OK] $file size=$(human_bytes "$actual")"
}

run_download() {
  check_commands
  resolve_token
  setup_curl_auth

  if [[ -e "$ROOT" && ! -d "$ROOT" ]]; then
    die "data root exists but is not a directory: $ROOT"
  fi
  mkdir -p "$ROOT/raw/grasp-anything-pp" "$ROOT/raw/grasp-anything"
  check_disk
  check_all_remote

  local entry repo file expected ctype
  for entry in "${FILES[@]}"; do
    IFS='|' read -r repo file expected ctype <<< "$entry"
    download_file "$repo" "$file" "$expected" "$ctype"
  done

  info ""
  info "All raw files downloaded to $ROOT/raw"
  info "Manifest: $ROOT/raw/download-manifest.tsv"
}

run_check() {
  check_commands
  resolve_token
  setup_curl_auth

  if [[ -e "$ROOT" && ! -d "$ROOT" ]]; then
    die "data root exists but is not a directory: $ROOT"
  fi

  info "[CHECK] mirror=$HF_MIRROR_BASE"
  info "[CHECK] root=$ROOT"
  info "[CHECK] token=present"
  check_disk
  check_all_remote
  info ""
  info "[CHECK] all mirror preflight checks passed"
}

main() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --check)
        MODE="check"
        shift
        ;;
      --download)
        MODE="download"
        shift
        ;;
      --root)
        [[ $# -ge 2 ]] || die "--root requires a directory"
        ROOT="$2"
        shift 2
        ;;
      --root=*)
        ROOT="${1#*=}"
        shift
        ;;
      --no-range-probe)
        RANGE_PROBE=0
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "unknown option: $1"
        ;;
    esac
  done

  if [[ -z "$ROOT" ]]; then
    ROOT="$DEFAULT_DATA_ROOT"
  fi
  if [[ "$ROOT" != /* ]]; then
    ROOT="$(pwd -P)/$ROOT"
  fi

  if [[ "$MODE" == "check" ]]; then
    run_check
  else
    run_download
  fi
}

main "$@"
