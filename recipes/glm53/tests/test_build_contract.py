"""Authenticate installed source and metadata while preserving native dependencies."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

RECIPE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "source_locked_image_labels", RECIPE / "source_locked_image_labels.py"
)
labels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labels)


@pytest.fixture
def lock():
    values = {
        "release.name": "serving-fixture",
        "release.version": "test",
        "runtime.base.image": "runtime@sha256:" + "a" * 64,
        "runtime.rootfs.layers": "2",
        "runtime.cache.fingerprint": "source-fixture",
        "runtime.cudagraph.mode": "FULL_AND_PIECEWISE",
        "runtime.lmcache.transfer": "engine-driven asynchronous shared memory",
        "runtime.scheduler.max-num-batched-tokens": "4096",
        "runtime.scheduler.prefill-compute-share": "0.4",
        "runtime.cuda.version": "13.3",
        "runtime.pytorch.version": "2.13.0",
        "model.repository": "target/fixture",
        "draft.repository": "draft/fixture",
        "draft.quantization": "MXFP8",
        "lmcache.version": "0.5.5.dev0",
    }
    for name in ("vllm", "b12x", "lmcache"):
        for key in ("commit", "tree", "package.tree"):
            values[f"{name}.{key}"] = f"{name}-{key}"
    for key in ("base.commit", "patch.sha256", "extension.sha256"):
        values[f"flashkda.{key}"] = key
    return values


def test_serving_labels_replace_incompatible_foundation_claims(lock):
    inherited = {
        "local-inference.vllm.composition": "different PR stack",
        "local-inference.cache.target-block-size": "512",
        "local-inference.glm53.mtp-backends": "moe:Humming",
        "local-inference.runtime.source-lock.sha256": "different-lock",
        "local-inference.scheduler.prefill-schedule-interval": "8",
        "org.opencontainers.image.version": "different-image",
    }
    overrides = labels.image_labels(lock, inherited, "b" * 64)
    assert overrides["local-inference.vllm.composition"] == ""
    assert overrides["local-inference.cache.target-block-size"] == ""
    assert "moe:Marlin" in overrides["local-inference.glm53.mtp-backends"]
    assert overrides["local-inference.runtime.source-lock.sha256"] == "b" * 64
    assert overrides["local-inference.scheduler.prefill-schedule-interval"] == "1"
    assert overrides["org.opencontainers.image.version"] == "test"
    assert overrides["local-inference.status"] == "implemented"
    assert all("qualified" not in value for value in overrides.values())


def test_unchanged_dependency_labels_are_preserved_by_inheritance(lock):
    dependency = "local-inference.flashinfer.commit"
    overrides = labels.image_labels(lock, {dependency: "source-ref"}, "b" * 64)
    assert dependency not in overrides
    for name in ("vllm", "b12x", "lmcache"):
        assert overrides[f"local-inference.{name}.commit"] == lock[f"{name}.commit"]


def test_replaced_flashinfer_artifact_does_not_inherit_another_source_ref(lock):
    for key in (
        "commit",
        "artifact.image",
        "python.wheel.sha256",
        "jit-cache.wheel.sha256",
    ):
        lock[f"flashinfer.{key}"] = key
    overrides = labels.image_labels(
        lock, {"local-inference.flashinfer.ref": "unrelated-source"}, "b" * 64
    )
    assert overrides["local-inference.flashinfer.ref"] == ""
    assert overrides["local-inference.flashinfer.commit"] == "commit"
    assert overrides["local-inference.flashinfer.artifact.image"] == "artifact.image"


def test_ambiguous_source_lock_is_rejected(tmp_path):
    source = tmp_path / "source.lock"
    source.write_text("vllm.commit=a\nvllm.commit=b\n")
    with pytest.raises(ValueError, match="Duplicate"):
        labels.read_lock(source)


def git(directory: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(directory), *arguments], text=True
    ).strip()


@pytest.fixture
def source_bundle(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "--quiet")
    git(source, "config", "user.name", "Source fixture")
    git(source, "config", "user.email", "source@example.invalid")
    (source / ".gitignore").write_text("*.so\n")
    (source / "module.py").write_text("value = 1\n")
    (source / "obsolete.py").write_text("value = 2\n")
    git(source, "add", ".")
    git(source, "commit", "--quiet", "-m", "Source fixture")
    destination = tmp_path / "installed source"
    subprocess.run(
        ["git", "clone", "--quiet", str(source), str(destination)], check=True
    )
    (destination / "extension.so").write_bytes(b"qualified native artifact")
    (source / "obsolete.py").unlink()
    (source / "module.py").write_text("value = 3\n")
    git(source, "add", "-A")
    git(source, "commit", "--quiet", "-m", "Complete source fixture")
    commit = git(source, "rev-parse", "HEAD")
    tree = git(source, "rev-parse", "HEAD^{tree}")
    bundle = tmp_path / "source.bundle"
    git(source, "bundle", "create", str(bundle), "HEAD")
    return destination, bundle, commit, tree


def test_installed_tree_matches_bundle_without_losing_native(source_bundle):
    destination, bundle, commit, tree = source_bundle
    for _ in range(2):
        subprocess.run(
            [
                "bash",
                str(RECIPE / "install_source_bundle.sh"),
                str(bundle),
                commit,
                tree,
                str(destination),
            ],
            check=True,
        )
        assert git(destination, "rev-parse", "HEAD") == commit
        assert git(destination, "status", "--porcelain") == ""
        assert (destination / "module.py").read_text() == "value = 3\n"
        assert not (destination / "obsolete.py").exists()
        assert (
            destination / "extension.so"
        ).read_bytes() == b"qualified native artifact"


@pytest.mark.parametrize("mismatch", ["commit", "tree"])
def test_lock_mismatch_does_not_replace_installed_sources(source_bundle, mismatch):
    destination, bundle, commit, tree = source_bundle
    before = git(destination, "rev-parse", "HEAD")
    if mismatch == "commit":
        commit = "0" * 40
    else:
        tree = "0" * 40
    result = subprocess.run(
        [
            "bash",
            str(RECIPE / "install_source_bundle.sh"),
            str(bundle),
            commit,
            tree,
            str(destination),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert git(destination, "rev-parse", "HEAD") == before
    assert (destination / "module.py").read_text() == "value = 1\n"
    assert (destination / "obsolete.py").read_text() == "value = 2\n"
