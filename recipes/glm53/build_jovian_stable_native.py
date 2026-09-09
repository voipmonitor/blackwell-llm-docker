"""Build the stable vLLM extension from a committed source snapshot."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vllm", "cutlass", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if git(args.vllm, "status", "--porcelain"):
        raise ValueError("Native compilation requires committed vLLM sources")
    if args.jobs < 1 or args.output.exists():
        raise ValueError("Use positive build jobs and an absent output directory")
    recipe = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="jovian-native-") as temporary:
        root = Path(temporary)
        archive = root / "source.tar"
        subprocess.run(
            ["git", "-C", str(args.vllm), "archive", "HEAD", f"--output={archive}"],
            check=True,
        )
        source = root / "source"
        source.mkdir()
        subprocess.run(["tar", "-xf", str(archive), "-C", str(source)], check=True)
        identity = {"vllm.commit": git(args.vllm, "rev-parse", "HEAD")}
        for key, path in (
            ("csrc.tree", "csrc"),
            ("cmake.tree", "cmake"),
            ("cmakelists.blob", "CMakeLists.txt"),
        ):
            identity[f"vllm.{key}"] = git(args.vllm, "rev-parse", f"HEAD:{path}")
        identity.update(
            {
                "cutlass.version": "4.4.2",
                "cuda.architecture": "120",
                "native.target": "_C_stable_libtorch",
            }
        )
        (source / "native-source.identity").write_text(
            "".join(f"{key}={value}\n" for key, value in identity.items())
        )
        subprocess.run(
            [
                "docker",
                "buildx",
                "build",
                "--progress",
                "plain",
                "--build-context",
                f"vllm_source={source}",
                "--build-context",
                f"cutlass_source={args.cutlass.resolve()}",
                "--build-arg",
                f"BUILD_JOBS={args.jobs}",
                "--output",
                f"type=local,dest={args.output.resolve()}",
                "-f",
                str(recipe / "Dockerfile.jovian-stable-native"),
                str(recipe),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
