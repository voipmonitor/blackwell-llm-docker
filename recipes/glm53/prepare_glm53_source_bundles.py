"""Freeze committed serving sources and build inputs for a flat Docker image."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vllm", "b12x", "lmcache", "native-artifact", "uv", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--release-name", default="jovian-judgement-community-source-locked"
    )
    parser.add_argument("--release-version", default="source-locked")
    args = parser.parse_args()
    args.output = args.output.resolve()
    roots = {
        name: getattr(args, name).resolve() for name in ("vllm", "b12x", "lmcache")
    }
    for name, root in roots.items():
        if git(root, "status", "--porcelain"):
            raise ValueError(f"{name} must have a clean committed source tree")
    args.output.mkdir(parents=True, exist_ok=False)
    lock = {
        "format": "local-inference-source-lock/v1",
        "implementation.status": "implemented",
        "qualification.contract": "See evidence for the resulting immutable image",
        "release.name": args.release_name,
        "release.version": args.release_version,
        "runtime.base.image": "voipmonitor/vllm@sha256:93ac5228f1cbde2182ca294d8b479259144742af2756a49ff207dd245429bf43",
        "runtime.cuda.version": "13.3",
        "runtime.pytorch.version": "2.13.0",
        "runtime.rootfs.layers": "2",
        "runtime.cudagraph.mode": "FULL_AND_PIECEWISE",
        "runtime.scheduler.max-num-batched-tokens": "4096",
        "runtime.scheduler.prefill-compute-share": "0.4",
        "runtime.nccl.channels": "16",
        "runtime.nccl.buffer.bytes": "2097152",
        "runtime.recurrent-checkpoint-policy": "auto",
        "runtime.lmcache.transfer": "engine-driven asynchronous shared memory",
        "runtime.lmcache.checkpoints": "atomic target and draft bundles for request_boundaries; independent chunks for explicit aligned",
        "model.repository": "local-inference-lab/GLM-5.3-Flash-NVFP4",
        "draft.repository": "local-inference-lab/GLM-5.3-Flash-DFlash2",
        "draft.quantization": "MXFP8",
    }
    for name, root in roots.items():
        revision = git(root, "rev-parse", "HEAD")
        lock[f"{name}.commit"] = revision
        lock[f"{name}.tree"] = git(root, "rev-parse", "HEAD^{tree}")
        lock[f"{name}.package.tree"] = git(root, "rev-parse", f"HEAD:{name}")
        bundle = args.output / f"{name}.bundle"
        subprocess.run(
            ["git", "-C", str(root), "bundle", "create", str(bundle), "HEAD"],
            check=True,
        )
        lock[f"{name}.bundle.sha256"] = digest(bundle)
    lock["lmcache.version"] = (
        f"0.5.5.dev0+glm53checkpoints.{lock['lmcache.commit'][:8]}"
    )
    patch = "cmake/external_projects/patches/flashkda-packed-checkpoints.patch"
    lock["flashkda.base.commit"] = (
        (args.native_artifact / "flashkda-base.commit").read_text().strip()
    )
    lock["flashkda.patch.sha256"] = digest(roots["vllm"] / patch)
    recorded_patch = (
        (args.native_artifact / "flashkda-patch.sha256").read_text().split()[0]
    )
    if recorded_patch != lock["flashkda.patch.sha256"]:
        raise ValueError(
            "Native artifact patch differs from the committed vLLM dependency"
        )
    lock["flashkda.extension.sha256"] = digest(
        args.native_artifact / "_flashkda_C.abi3.so"
    )
    docker = Path(__file__).resolve().parent
    inputs = (
        "Dockerfile.glm53-cache-contracts",
        "build_glm53_cache_contract_image.sh",
        "source_locked_image_labels.py",
        "install_source_bundle.sh",
        "install_glm53_source_locked.sh",
        "serve-glm53-flash-nvfp4-dflash2.sh",
        "serve-glm53-flash-nvfp4-dflash2-scheduler-qos.sh",
        "serve-glm53-flash-lmcache.sh",
        "serve-glm53-flash-lmcache-cache-complete.sh",
        "serve-glm53-flash-cache-complete.sh",
        "glm53_checkpoint_identity.py",
        "glm53-r18-lmcache-runtime-requirements.txt",
    )
    for filename in inputs:
        lock[f"input.{filename}.sha256"] = digest(docker / filename)
    shutil.copy2(args.uv, args.output / "uv")
    lock["build.uv.sha256"] = digest(args.output / "uv")
    lock["runtime.cache.fingerprint"] = (
        f"cu133-torch213-glm53-vllm{lock['vllm.package.tree'][:8]}"
        f"-b12x{lock['b12x.package.tree'][:8]}-lmcache{lock['lmcache.package.tree'][:8]}"
        f"-flashkda{lock['flashkda.extension.sha256'][:8]}"
    )
    target = args.output / "source.lock"
    target.write_text("".join(f"{key}={value}\n" for key, value in lock.items()))
    print(f"SOURCE_LOCK_SHA256={digest(target)}")
    print(f"CACHE_FINGERPRINT={lock['runtime.cache.fingerprint']}")


if __name__ == "__main__":
    main()
