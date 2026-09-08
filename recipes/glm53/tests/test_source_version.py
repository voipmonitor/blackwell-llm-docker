"""Generated package versions must identify installed serving Python sources."""

import importlib.util
from pathlib import Path

import pytest

RECIPE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "install_vllm_source_version", RECIPE / "install_vllm_source_version.py"
)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


@pytest.fixture
def source(tmp_path):
    (tmp_path / "vllm").mkdir()
    (tmp_path / "vllm.egg-info").mkdir()
    (tmp_path / "vllm.egg-info/PKG-INFO").write_text(
        "Name: vllm\nVersion: 0.0.0\nSummary: Serving library\n"
    )
    return tmp_path


def test_python_and_distribution_versions_agree(source):
    version = "0.26.1rc0+glm53.dspark.vllmabcdef01"
    installer.install_version(source, version)
    namespace = {}
    exec((source / "vllm/_version.py").read_text(), namespace)
    assert namespace["__version__"] == version
    assert namespace["version"] == version
    assert namespace["version_tuple"] == (0, 26, 1, "rc0", "glm53.dspark.vllmabcdef01")
    assert (source / "vllm.egg-info/PKG-INFO").read_text() == (
        f"Name: vllm\nVersion: {version}\nSummary: Serving library\n"
    )


def test_ambiguous_distribution_metadata_is_rejected_before_writing(source):
    metadata = source / "vllm.egg-info/PKG-INFO"
    metadata.write_text("Version: 1\nVersion: 2\n")
    with pytest.raises(ValueError, match="exactly one"):
        installer.install_version(source, "0.26.1rc0+glm53.dspark")
    assert not (source / "vllm/_version.py").exists()
    assert metadata.read_text() == "Version: 1\nVersion: 2\n"


def test_unidentified_source_version_is_rejected(source):
    with pytest.raises(ValueError, match="source identity"):
        installer.install_version(source, "0.26.1rc0")
    assert not (source / "vllm/_version.py").exists()
