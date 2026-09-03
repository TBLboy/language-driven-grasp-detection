"""Small HTTP Range helpers for inspecting large remote zip archives.

The published Grasp-Anything++ zips have central directories of several
hundred MB, so this module avoids downloading them.  A local tail snapshot of
the central directory is enough to recover individual member offsets.
"""

from __future__ import annotations

import argparse
import struct
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path


CDH_SIG = b"PK\x01\x02"
CD_HEADER = struct.Struct("<IHHHHHHIIIHHHHHII")
LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")


@dataclass(frozen=True)
class ZipMember:
    name: str
    local_offset: int
    compress_size: int
    file_size: int
    compress_type: int
    crc: int


def _read_exact(url: str, offset: int, length: int) -> bytes:
    if length <= 0:
        return b""
    req = urllib.request.Request(
        url,
        headers={"Range": f"bytes={offset}-{offset + length - 1}"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    if len(data) != length:
        raise RuntimeError(
            f"short range read at {offset}: expected {length}, got {len(data)}"
        )
    return data


def _zip64_values(extra: bytes) -> list[int]:
    values: list[int] = []
    pos = 0
    while pos + 4 <= len(extra):
        field_id, field_size = struct.unpack_from("<HH", extra, pos)
        pos += 4
        field_data = extra[pos : pos + field_size]
        pos += field_size
        if field_id == 0x0001:
            offset = 0
            while offset + 8 <= len(field_data):
                values.append(struct.unpack_from("<Q", field_data, offset)[0])
                offset += 8
    return values


def parse_tail_index(tail_path: str | Path) -> dict[str, ZipMember]:
    """Parse every central-directory record present in a local zip tail."""

    data = Path(tail_path).read_bytes()
    index: dict[str, ZipMember] = {}

    for i in range(len(data) - CD_HEADER.size):
        if data[i : i + 4] != CDH_SIG:
            continue
        (
            _sig,
            _version_made,
            _version_needed,
            _flags,
            compress_type,
            _mod_time,
            _mod_date,
            crc,
            compress_size,
            file_size,
            name_len,
            extra_len,
            _comment_len,
            _disk_start,
            _internal_attr,
            _external_attr,
            local_offset,
        ) = CD_HEADER.unpack_from(data, i)

        name = data[i + 46 : i + 46 + name_len].decode("utf-8", errors="replace")
        extra = data[i + 46 + name_len : i + 46 + name_len + extra_len]
        values = _zip64_values(extra)
        value_pos = 0

        if file_size == 0xFFFFFFFF and value_pos < len(values):
            file_size = values[value_pos]
            value_pos += 1
        if compress_size == 0xFFFFFFFF and value_pos < len(values):
            compress_size = values[value_pos]
            value_pos += 1
        if local_offset == 0xFFFFFFFF and value_pos < len(values):
            local_offset = values[value_pos]
            value_pos += 1

        index[name] = ZipMember(
            name=name,
            local_offset=local_offset,
            compress_size=compress_size,
            file_size=file_size,
            compress_type=compress_type,
            crc=crc,
        )

    return index


def extract_member(
    url: str,
    member: ZipMember,
    output: str | Path,
) -> bytes:
    """Download one member using the offset recovered from a tail snapshot."""

    header = _read_exact(url, member.local_offset, LOCAL_HEADER.size)
    (
        sig,
        _version_needed,
        _flags,
        _local_compress_type,
        _mod_time,
        _mod_date,
        _local_crc,
        _local_compress_size,
        _local_file_size,
        name_len,
        extra_len,
    ) = LOCAL_HEADER.unpack(header)

    if sig != 0x04034B50:
        raise RuntimeError(
            f"bad local header at {member.local_offset} for {member.name}"
        )

    local_name = _read_exact(
        url,
        member.local_offset + LOCAL_HEADER.size,
        name_len,
    ).decode("utf-8", errors="replace")
    if local_name != member.name:
        raise RuntimeError(
            f"local name mismatch: expected {member.name}, got {local_name}"
        )

    data_offset = member.local_offset + LOCAL_HEADER.size + name_len + extra_len
    compressed = _read_exact(url, data_offset, member.compress_size)

    if member.compress_type == 0:
        payload = compressed
    elif member.compress_type == 8:
        payload = zlib.decompress(compressed, -15)
    else:
        raise NotImplementedError(
            f"unsupported compression type {member.compress_type}"
        )

    if len(payload) != member.file_size:
        raise RuntimeError(
            f"size mismatch for {member.name}: expected {member.file_size}, "
            f"got {len(payload)}"
        )

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_bytes(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--tail", required=True)
    parser.add_argument("--member", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tail_index = parse_tail_index(args.tail)
    member = tail_index.get(args.member)
    if member is None:
        raise SystemExit(f"member not found in tail index: {args.member}")
    extract_member(args.url, member, args.output)
    print(f"extracted {args.member} -> {args.output}")


if __name__ == "__main__":
    main()
