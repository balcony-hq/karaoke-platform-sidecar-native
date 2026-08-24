#!/usr/bin/env python3
"""Extract and validate models/model.onnx from a sidecar tar.gz archive."""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    options = parse_args()
    with tarfile.open(options.archive, mode="r:gz") as archive:
        try:
            member = archive.getmember("models/model.onnx")
        except KeyError as error:
            raise FileNotFoundError("archive does not contain models/model.onnx") from error
        if not member.isfile() or member.issym() or member.islnk():
            raise ValueError("models/model.onnx is not a regular archive file")
        source = archive.extractfile(member)
        if source is None:
            raise ValueError("could not read models/model.onnx from archive")
        options.output.parent.mkdir(parents=True, exist_ok=True)
        with options.output.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    print(f"Extracted {options.output}")


if __name__ == "__main__":
    main()
