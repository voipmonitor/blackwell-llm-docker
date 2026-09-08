"""Identify installed Python sources independently of native build provenance."""

from __future__ import annotations

import argparse
from pathlib import Path

from packaging.version import Version


def install_version(source: Path, version: str) -> None:
    parsed = Version(version)
    if parsed.local is None:
        raise ValueError("Serving version must include a source identity")
    metadata = source / "vllm.egg-info/PKG-INFO"
    lines = metadata.read_text().splitlines(keepends=True)
    if sum(line.startswith("Version: ") for line in lines) != 1:
        raise ValueError("Distribution metadata must have exactly one Version field")
    prerelease = (f"{parsed.pre[0]}{parsed.pre[1]}",) if parsed.pre else ()
    version_tuple = (*parsed.release, *prerelease, parsed.local)
    (source / "vllm/_version.py").write_text(
        f"__version__ = version = {version!r}\n"
        f"__version_tuple__ = version_tuple = {version_tuple!r}\n"
    )
    metadata.write_text(
        "".join(
            f"Version: {version}\n" if line.startswith("Version: ") else line
            for line in lines
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("version")
    args = parser.parse_args()
    install_version(args.source, args.version)
