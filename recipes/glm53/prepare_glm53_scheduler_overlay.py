"""Freeze a Python-only scheduler overlay over the immutable FP8 serving image."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from prepare_glm53_source_bundles import digest, git
from source_locked_image_labels import read_lock

PARENT = "voipmonitor/vllm@sha256:f5f121e37fd2afbb6f8f036e7eb627435cfb736de0a4420306dc2a25b6631669"
PARENT_LOCK_SHA = "15a9a649559830822cb943ea0c3c6a644c8b69c9e54d7be3e265801851843932"
INPUTS = (
    "Dockerfile.glm53-scheduler-overlay",
    "build_glm53_scheduler_overlay.sh",
    "prepare_glm53_scheduler_overlay.py",
    "prepare_glm53_source_bundles.py",
    "install_glm53_scheduler_overlay.py",
    "install_source_bundle.sh",
    "source_locked_image_labels.py",
    "serve-glm53-flash-nvfp4-dflash2.sh",
    "serve-glm53-flash-nvfp4-dflash2-scheduler-qos.sh",
    "serve-glm53-flash-cache-complete.sh",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vllm", "parent-lock", "uv", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--release-name", required=True)
    parser.add_argument("--release-version", required=True)
    args = parser.parse_args()
    if digest(args.parent_lock) != PARENT_LOCK_SHA:
        raise ValueError("The overlay requires the authenticated R28 source lock")
    root = args.vllm.resolve()
    if git(root, "status", "--porcelain"):
        raise ValueError("vLLM must be a clean committed source tree")
    lock = read_lock(args.parent_lock)
    parent_commit = lock["vllm.commit"]
    git(root, "merge-base", "--is-ancestor", parent_commit, "HEAD")
    changed = git(root, "diff", "--name-only", parent_commit, "HEAD").splitlines()
    allowed = {
        "vllm/config/scheduler.py",
        "vllm/v1/core/sched/prefill_interleave.py",
        "vllm/v1/core/sched/scheduler.py",
        "vllm/v1/engine/core.py",
        "tests/v1/core/test_prefill_compute_share_scheduler.py",
        "tests/v1/core/test_scheduler.py",
        "tests/v1/engine/test_compute_fairness_feedback.py",
    }
    if not changed or set(changed) - allowed:
        raise ValueError(f"Changes exceed the Python scheduler contract: {changed}")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = args.output / "vllm.bundle"
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "bundle",
            "create",
            str(bundle),
            "HEAD",
            f"^{parent_commit}",
        ],
        check=True,
    )
    lock.update(
        {
            "release.name": args.release_name,
            "release.version": args.release_version,
            "runtime.parent.release-image": PARENT,
            "runtime.parent.source-lock.sha256": PARENT_LOCK_SHA,
            "runtime.rootfs.layers": "3",
            "runtime.default.reasoning-effort": "high",
            "runtime.scheduler.max-parallel-prefills": "1",
            "runtime.scheduler.auto-max-parallel-prefills": "4",
            "vllm.parent.commit": parent_commit,
            "vllm.commit": git(root, "rev-parse", "HEAD"),
            "vllm.tree": git(root, "rev-parse", "HEAD^{tree}"),
            "vllm.package.tree": git(root, "rev-parse", "HEAD:vllm"),
            "vllm.bundle.sha256": digest(bundle),
            "vllm.bundle.mode": "incremental over parent image; complete installed Git history",
        }
    )
    lock["vllm.version"] = (
        f"0.26.1rc0+glm53.{args.release_version}.vllm{lock['vllm.commit'][:8]}"
    )
    recipe = Path(__file__).resolve().parent
    for filename in INPUTS:
        lock[f"input.{filename}.sha256"] = digest(recipe / filename)
    shutil.copy2(args.uv, args.output / "uv")
    lock["build.uv.sha256"] = digest(args.output / "uv")
    lock["runtime.cache.fingerprint"] = (
        f"cu133-torch213-glm53-vllm{lock['vllm.package.tree'][:8]}"
        f"-b12x{lock['b12x.package.tree'][:8]}-lmcache{lock['lmcache.package.tree'][:8]}"
        f"-flashkda{lock['flashkda.extension.sha256'][:8]}"
    )
    (args.output / "source.lock").write_text(
        "".join(f"{key}={value}\n" for key, value in lock.items())
    )
    print(f"SOURCE_LOCK_SHA256={digest(args.output / 'source.lock')}")


if __name__ == "__main__":
    main()
