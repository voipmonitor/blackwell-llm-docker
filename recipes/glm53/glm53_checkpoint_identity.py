#!/usr/bin/env python3
"""Resolve immutable model identities before enabling persistent checkpoints.

Repository names are resolved once to a Hugging Face commit; the caller must
pass the returned revision to the model loader. Local checkpoints are identified
by their weight and configuration bytes, not by their directory names. Local
model files must remain unchanged for the lifetime of the serving process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def local_checkpoint_identity(directory: Path) -> str:
    """Hash local safetensors weights and model/tokenizer configuration.

    JSON, Jinja and tokenizer assets are included conservatively. Reading does
    not modify the checkpoint. A checkpoint that changes during this operation
    is rejected; callers must also keep it immutable during model loading and
    serving. Missing weights or unsupported weight formats are errors.
    """
    directory = directory.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError(f"Checkpoint must be a directory: {directory}")

    def files() -> list[Path]:
        return sorted(
            path
            for path in directory.iterdir()
            if path.is_file()
            and (
                path.suffix in {".safetensors", ".json", ".jinja", ".model"}
                or path.name in {"vocab.txt", "merges.txt", "added_tokens.txt"}
            )
        )

    selected = files()
    if not any(path.suffix == ".safetensors" for path in selected):
        raise ValueError("Persistent checkpoint identity requires safetensors weights")
    if directory / "config.json" not in selected:
        raise ValueError("Persistent checkpoint identity requires config.json")
    if any(directory.glob("*.bin")) or any(directory.glob("*.gguf")):
        raise ValueError("Ambiguous checkpoint: non-safetensors weights are present")

    def content_metadata(stat: os.stat_result) -> tuple[int, ...]:
        # Reading may update atime without changing the model. Inode, size,
        # mtime and ctime detect replacement or writes without rejecting reads.
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    before = {path: path.stat() for path in selected}
    manifest: list[tuple[str, int, str]] = []
    for path in selected:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest.append((path.name, before[path].st_size, digest))
    if files() != selected or any(
        content_metadata(path.stat()) != content_metadata(before[path])
        for path in selected
    ):
        raise ValueError("Checkpoint files changed while their identity was computed")
    encoded = json.dumps(manifest, separators=(",", ":")).encode()
    return hashlib.sha256(b"glm53-local-checkpoint-v1\0" + encoded).hexdigest()


def resolve_checkpoint(model: str, revision: str | None = None) -> dict[str, str]:
    """Return a content identity and, for a Hub repository, a pinned revision.

    Local files are hashed directly, including symlink targets. For a Hub model,
    config.json is resolved through the installed Hugging Face cache client so
    its standard authentication, offline and cache settings remain effective.
    The returned revision must be used for every weight/configuration download.
    """
    path = Path(model).expanduser()
    if path.exists():
        return {"identity": local_checkpoint_identity(path), "revision": ""}
    if path.is_absolute() or model.startswith(("./", "../", "~")):
        raise ValueError(f"Local checkpoint directory does not exist: {model}")

    from huggingface_hub import hf_hub_download

    config_path = Path(
        hf_hub_download(model, "config.json", revision=revision or "main")
    )
    # The Hub cache client returns snapshots/<commit>/config.json. Do not
    # resolve its symlink: the target blob path no longer contains the commit.
    commit = config_path.parent.name
    if (
        config_path.parent.parent.name != "snapshots"
        or len(commit) != 40
        or any(char not in "0123456789abcdef" for char in commit)
    ):
        raise ValueError("Hugging Face did not return an immutable snapshot path")
    return {"identity": commit, "revision": commit}


def resolve_serving_identity(
    model: str,
    revision: str | None,
    draft_model: str | None,
    draft_revision: str | None,
    speculation: str,
    source_lock: Path,
) -> dict[str, object]:
    """Identify weights and a build-verified serving source lock.

    ``speculation`` is none, mtp, or dflash. MTP shares target checkpoint weights;
    DFlash requires a separate checkpoint. The source lock must describe the
    image's installed code and native artifacts; source-code mounts are outside
    this launcher's immutable-image contract.
    """
    if speculation not in {"none", "mtp", "dflash"}:
        raise ValueError("Speculation must be none, mtp, or dflash")
    if speculation == "dflash" and not draft_model:
        raise ValueError("DFlash requires a draft model")
    lock_bytes = source_lock.read_bytes()
    if not lock_bytes.startswith(b"format=local-inference-source-lock/v1\n"):
        raise ValueError("Unrecognized serving source-lock format")
    target = resolve_checkpoint(model, revision)
    draft = {"identity": "", "revision": ""}
    if speculation == "mtp":
        draft = target
    elif speculation == "dflash":
        assert draft_model is not None
        draft = resolve_checkpoint(draft_model, draft_revision)
    return {
        "checkpoint_identity": {
            "target_revision": target["identity"],
            "draft_revision": draft["identity"],
            "source_revision": hashlib.sha256(lock_bytes).hexdigest(),
        },
        "model_revision": target["revision"],
        "draft_model_revision": draft["revision"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--draft-model")
    parser.add_argument("--draft-revision")
    parser.add_argument(
        "--speculation", choices=("none", "mtp", "dflash"), required=True
    )
    parser.add_argument(
        "--source-lock", type=Path, default=Path("/opt/glm53-flash/source.lock")
    )
    args = parser.parse_args()
    result = resolve_serving_identity(
        args.model,
        args.revision,
        args.draft_model,
        args.draft_revision,
        args.speculation,
        args.source_lock,
    )
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
