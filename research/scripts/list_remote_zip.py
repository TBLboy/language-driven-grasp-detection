#!/usr/bin/env python
"""List members of a remote zip by fetching only its central directory.

This is a research utility for inspecting large Hugging Face dataset zips
without downloading every compressed member.
"""

from __future__ import annotations

import argparse
import struct
import time
import urllib.request
from pathlib import Path


EOCD_SIG = b"PK\x05\x06"
CDH_SIG = b"PK\x01\x02"


class RangeReader:
    def __init__(self, url: str, timeout: int = 120):
        self.url = url
        self.timeout = timeout
        self.size = self._head_size()

    def _head_size(self) -> int:
        req = urllib.request.Request(self.url, method="HEAD")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return int(resp.headers["Content-Length"])

    def read(self, offset: int, length: int) -> bytes:
        if offset < 0 or length <= 0:
            raise ValueError("invalid range")
        end = min(self.size - 1, offset + length - 1)
        req = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={offset}-{end}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = resp.read()
        if len(data) != end - offset + 1:
            raise RuntimeError(
                f"short range read at {offset}: expected {end - offset + 1}, got {len(data)}"
            )
        return data


def find_eocd(tail: bytes, file_size: int) -> tuple[int, dict]:
    for i in range(len(tail) - 22, -1, -1):
        if tail[i : i + 4] != EOCD_SIG:
            continue
        fields = struct.unpack_from("<IHHHHIIH", tail, i)
        return file_size - len(tail) + i, {
            "disk_number": fields[1],
            "cd_start_disk": fields[2],
            "entries_on_disk": fields[3],
            "total_entries": fields[4],
            "cd_size": fields[5],
            "cd_offset": fields[6],
            "comment_len": fields[7],
        }
    raise RuntimeError("EOCD not found")


def list_members(url: str, output: Path) -> list[str]:
    reader = RangeReader(url)
    tail_len = min(reader.size, 65557)
    tail = reader.read(reader.size - tail_len, tail_len)
    eocd_offset, eocd = find_eocd(tail, reader.size)
    cd_size = eocd["cd_size"]
    cd_offset = eocd["cd_offset"]
    print(
        f"{url.split('/')[-1]}: size={reader.size}, cd_offset={cd_offset}, "
        f"cd_size={cd_size}, entries={eocd['total_entries']}, eocd_offset={eocd_offset}"
    )

    if cd_offset + cd_size > reader.size:
        # Zip64 fallback is intentionally explicit: inspect the original file
        # before assuming 32-bit offsets apply.
        raise RuntimeError("zip64 central directory detected; not supported yet")

    central = reader.read(cd_offset, cd_size)
    current = 0
    names: list[str] = []
    while current + 46 <= len(central):
        if central[current : current + 4] != CDH_SIG:
            break
        (
            _sig,
            _ver_made,
            _ver_needed,
            _flags,
            _compression,
            _mtime,
            _mdate,
            _crc,
            _csize,
            _usize,
            name_len,
            extra_len,
            comment_len,
            _disk_start,
            _internal_attr,
            _external_attr,
            _local_offset,
        ) = struct.unpack_from("<IHHHHHHIIIHHHHHII", central, current)
        name_bytes = central[current + 46 : current + 46 + name_len]
        names.append(name_bytes.decode("utf-8", errors="replace"))
        current += 46 + name_len + extra_len + comment_len

    if len(names) != eocd["total_entries"]:
        raise RuntimeError(
            f"parsed {len(names)} entries, expected {eocd['total_entries']}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"saved {len(names)} names to {output}")
    print(f"first: {names[:3]}")
    print(f"last: {names[-3:]}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    started = time.time()
    names = list_members(args.url, Path(args.output))
    print(f"elapsed: {time.time() - started:.2f}s")
    _ = names


if __name__ == "__main__":
    main()
