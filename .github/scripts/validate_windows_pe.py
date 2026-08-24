#!/usr/bin/env python3
"""Validate the PE machine type of every staged Windows executable and DLL."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path


MACHINE_TYPES = {
    "x86": 0x014C,
    "amd64": 0x8664,
    "arm64": 0xAA64,
}
MACHINE_NAMES = {value: name for name, value in MACHINE_TYPES.items()}
PE_SIGNATURE = b"PE\0\0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="staged bundle directory")
    parser.add_argument(
        "--machine",
        choices=sorted(MACHINE_TYPES),
        default="amd64",
        help="required PE machine type (default: amd64)",
    )
    return parser.parse_args()


def read_machine(path: Path) -> int:
    with path.open("rb") as handle:
        dos_header = handle.read(64)
        if len(dos_header) < 64 or dos_header[:2] != b"MZ":
            raise ValueError("missing DOS header")

        pe_offset = struct.unpack_from("<I", dos_header, 0x3C)[0]
        handle.seek(pe_offset)
        coff_header = handle.read(6)
        if len(coff_header) < 6 or coff_header[:4] != PE_SIGNATURE:
            raise ValueError("missing PE signature")
        return struct.unpack_from("<H", coff_header, 4)[0]


def staged_pe_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".dll", ".exe"}
    )


def main() -> None:
    options = parse_args()
    root = options.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"staged bundle directory not found: {root}")

    files = staged_pe_files(root)
    if not files:
        raise ValueError(f"no Windows .exe or .dll files found under {root}")

    expected = MACHINE_TYPES[options.machine]
    failures: list[str] = []
    for path in files:
        try:
            actual = read_machine(path)
        except (OSError, ValueError, struct.error) as error:
            failures.append(f"{path}: invalid PE ({error})")
            continue
        if actual != expected:
            actual_name = MACHINE_NAMES.get(actual, f"0x{actual:04x}")
            failures.append(
                f"{path}: machine is {actual_name}, expected {options.machine}"
            )

    if failures:
        raise ValueError("\n".join(failures))

    print(f"Validated {len(files)} Windows PE files as {options.machine}: {root}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"validate_windows_pe: {error}", file=sys.stderr)
        raise SystemExit(1)
