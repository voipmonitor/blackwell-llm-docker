"""Install authenticated Python sources without changing native dependencies."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from prepare_glm53_source_bundles import digest, git
from prepare_glm53_scheduler_overlay import INPUTS, PARENT_LOCK_SHA
from source_locked_image_labels import read_lock


def main() -> None:
    recipe = Path(__file__).resolve().parent
    inputs = Path("/inputs")
    installed = Path("/opt/glm53-flash/source.lock")
    lock = read_lock(inputs / "source.lock")
    assert digest(installed) == PARENT_LOCK_SHA
    assert digest(inputs / "source.lock") == os.environ["SOURCE_LOCK_SHA256"]
    for filename in INPUTS:
        assert digest(recipe / filename) == lock[f"input.{filename}.sha256"]
    assert digest(inputs / "vllm.bundle") == lock["vllm.bundle.sha256"]
    vllm = Path("/opt/glm53-flash/vllm")
    flashkda = vllm / "vllm/_flashkda_C.abi3.so"
    assert digest(flashkda) == lock["flashkda.extension.sha256"]
    assert git(vllm, "rev-parse", "HEAD") == lock["vllm.parent.commit"]
    assert not git(vllm, "status", "--porcelain")
    subprocess.run(
        [
            "bash",
            str(recipe / "install_source_bundle.sh"),
            str(inputs / "vllm.bundle"),
            lock["vllm.commit"],
            lock["vllm.tree"],
            str(vllm),
        ],
        check=True,
    )
    destinations = {
        "serve-glm53-flash-nvfp4-dflash2.sh": "/usr/local/libexec/serve-glm53-flash-nvfp4-dflash2.sh",
        "serve-glm53-flash-nvfp4-dflash2-scheduler-qos.sh": "/usr/local/bin/serve-glm53-flash-nvfp4-dflash2.sh",
        "serve-glm53-flash-cache-complete.sh": "/usr/local/bin/serve-glm53-flash.sh",
    }
    for source, destination in destinations.items():
        shutil.copyfile(recipe / source, destination)
        Path(destination).chmod(0o755)
    # Generated distribution metadata must describe the installed source, not
    # the unrelated source revision used to compile unchanged native artifacts.
    version = lock["vllm.version"]
    (vllm / "vllm/_version.py").write_text(
        f"__version__ = version = {version!r}\n"
        f"__version_tuple__ = version_tuple = (0, 26, 1, 'rc0', {version.split('+')[1]!r})\n"
    )
    metadata = vllm / "vllm.egg-info/PKG-INFO"
    lines = metadata.read_text().splitlines(keepends=True)
    assert sum(line.startswith("Version: ") for line in lines) == 1
    metadata.write_text(
        "".join(
            f"Version: {version}\n" if line.startswith("Version: ") else line
            for line in lines
        )
    )
    shutil.copyfile(inputs / "source.lock", installed)
    assert digest(flashkda) == lock["flashkda.extension.sha256"]
    for name, source in (
        ("vllm", vllm),
        ("b12x", Path("/opt/glm53-flash/b12x")),
        ("lmcache", Path("/opt/lmcache/source")),
    ):
        assert git(source, "rev-parse", "HEAD") == lock[f"{name}.commit"]
        assert git(source, "write-tree") == lock[f"{name}.tree"]
        assert not git(source, "status", "--porcelain")


if __name__ == "__main__":
    main()
