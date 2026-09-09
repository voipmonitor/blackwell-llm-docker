"""Check DS4 launcher sampling precedence without loading model weights."""

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

RECIPE = Path(__file__).resolve().parents[1]


@pytest.fixture
def launch(tmp_path):
    stub = tmp_path / "argv.sh"
    stub.write_text('#!/bin/bash\nprintf "%s\\0" "$@"\n')
    stub.chmod(0o755)
    wrapper = tmp_path / "serve.sh"
    source = (RECIPE / "serve-ds4-jovian.sh").read_text()
    wrapper.write_text(
        source.replace(
            "exec /usr/local/bin/lmcache-mp-wrapper.sh",
            f"exec {shlex.quote(str(stub))}",
        )
    )

    def run(arguments=(), environment=None):
        result = subprocess.run(
            ["bash", str(wrapper), *arguments],
            env={"PATH": os.environ["PATH"], **(environment or {})},
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.rstrip("\0").split("\0")[1:]

    return run


@pytest.mark.parametrize("variant", ["text", "vision"])
def test_agent_profile_uses_publisher_sampling(launch, variant):
    arguments = launch(environment={"DS4_MODEL_VARIANT": variant})
    assert arguments[0] == "--override-generation-config"
    assert json.loads(arguments[1]) == {"temperature": 1.0, "top_p": 0.95}


@pytest.mark.parametrize(
    "arguments,environment",
    [
        (["--generation-config", "vllm"], {}),
        (["--override-generation-config.top_p=0.8"], {}),
        (['--override-generation-config={"top_p":0.8}'], {}),
        (["--config", "/operator/serve.yaml"], {}),
        ([], {"EXTRA_VLLM_ARGS": '--override-generation-config={"top_p":0.8}'}),
        ([], {"EXTRA_VLLM_ARGS": "--generation_config vllm"}),
    ],
)
def test_operator_sampling_is_authoritative(launch, arguments, environment):
    assert launch(arguments, environment) == arguments
