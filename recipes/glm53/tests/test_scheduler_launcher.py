"""Validate scheduler environment mapping without loading models or using GPUs."""

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

RECIPE = Path(__file__).resolve().parents[1]
OPTIONS = {
    "PREFILL_COMPUTE_SHARE": "--prefill-compute-share",
    "PREFILL_COMPUTE_HALF_LIFE": "--prefill-compute-half-life",
    "MAX_PARALLEL_PREFILLS": "--max-parallel-prefills",
    "PREFILL_POLICY": "--prefill-policy",
    "DECODE_REFILL_TARGET": "--decode-refill-target",
}


@pytest.fixture
def launch(tmp_path):
    stub = tmp_path / "argv.sh"
    stub.write_text('#!/bin/bash\nprintf "%s\\0" "$@"\n')
    stub.chmod(0o755)
    wrapper = tmp_path / "scheduler.sh"
    source = (RECIPE / "serve-glm53-flash-nvfp4-dflash2-scheduler-qos.sh").read_text()
    wrapper.write_text(
        source.replace(
            "readonly base_launcher=/usr/local/libexec/serve-glm53-flash-nvfp4-dflash2.sh",
            f"readonly base_launcher={shlex.quote(str(stub))}",
        )
    )

    def run(environment=None, arguments=(), *, base=False):
        env = {
            "PATH": os.environ["PATH"],
            "CUDAGRAPH_CAPTURE_SIZES": "none",
            "PREFILL_SCHEDULE_INTERVAL": "1",
            **(environment or {}),
        }
        path = wrapper
        if base:
            path = RECIPE / "serve-glm53-flash-nvfp4-dflash2.sh"
            env["DRY_RUN"] = "1"
        return subprocess.run(
            ["bash", str(path), *arguments],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    return run


def argv(result):
    assert result.returncode == 0, result.stderr
    return [value for value in result.stdout.split("\0") if value]


def test_absent_controls_preserve_native_defaults(launch):
    assert argv(launch()) == []


def test_community_defaults_do_not_enable_interleaving(launch):
    assert argv(
        launch({"FAIRNESS_ENGINE": "compute_share", "PREFILL_COMPUTE_SHARE": "0.4"})
    ) == [
        "--prefill-compute-share",
        "0.4",
    ]


def test_glm_sampling_uses_publisher_defaults_without_checkpoint_metadata(launch):
    result = launch(base=True)
    assert result.returncode == 0, result.stderr
    arguments = shlex.split(result.stdout)[3:]
    option = "--override-generation-config"
    assert arguments.count(option) == 1
    assert json.loads(arguments[arguments.index(option) + 1]) == {
        "temperature": 1.0,
        "top_p": 0.95,
    }


@pytest.mark.parametrize(
    "arguments",
    [
        ["--override-generation-config", '{"top_p":0.8}'],
        ['--override-generation-config={"top_p":0.8}'],
        ["--override-generation-config.top_p", "0.8"],
        ["--override_generation_config.top_p=0.8"],
        ["--generation-config", "vllm"],
        ["--generation-config=/operator/model-defaults"],
        ["--config", "/operator/serve.yaml"],
    ],
)
def test_explicit_generation_policy_is_not_overwritten(launch, arguments):
    result = launch(arguments=arguments, base=True)
    assert result.returncode == 0, result.stderr
    rendered = shlex.split(result.stdout)[3:]
    assert rendered[-len(arguments) :] == arguments
    assert '{"temperature":1.0,"top_p":0.95}' not in rendered


def test_all_environment_controls_are_forwarded_once(launch):
    values = dict(zip(OPTIONS, ["auto", "responsive", "auto", "decode-aware", "auto"]))
    arguments = argv(launch(values))
    for environment, option in OPTIONS.items():
        assert arguments.count(option) == 1
        assert arguments[arguments.index(option) + 1] == values[environment]


@pytest.mark.parametrize("equals", [False, True])
def test_cli_overrides_invalid_environment_values(launch, equals):
    values = dict(zip(OPTIONS.values(), ["auto", "smooth", "4", "round-robin", "2"]))
    arguments = []
    for option, value in values.items():
        arguments.extend([f"{option}={value}"] if equals else [option, value])
    assert argv(launch({name: "invalid" for name in OPTIONS}, arguments)) == arguments


@pytest.mark.parametrize(
    "half_life", ["smooth", "responsive", "0.5", "2", "1e-3", "+2."]
)
def test_auto_half_life(launch, half_life):
    assert "--prefill-compute-half-life" in argv(
        launch(
            {
                "PREFILL_COMPUTE_SHARE": "auto",
                "PREFILL_COMPUTE_HALF_LIFE": half_life,
            }
        )
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("PREFILL_COMPUTE_SHARE", "0"),
        ("PREFILL_COMPUTE_SHARE", "1"),
        ("PREFILL_COMPUTE_SHARE", "nan"),
        ("PREFILL_COMPUTE_SHARE", "0.4junk"),
        ("PREFILL_COMPUTE_SHARE", "-0.4"),
        ("PREFILL_COMPUTE_HALF_LIFE", "0"),
        ("PREFILL_COMPUTE_HALF_LIFE", "inf"),
        ("PREFILL_COMPUTE_HALF_LIFE", "1e9999"),
        ("PREFILL_COMPUTE_HALF_LIFE", "garbage"),
        ("MAX_PARALLEL_PREFILLS", "0"),
        ("MAX_PARALLEL_PREFILLS", "1.5"),
        ("DECODE_REFILL_TARGET", "-1"),
        ("PREFILL_POLICY", "fastest"),
    ],
)
def test_invalid_controls_fail_before_model_launch(launch, name, value):
    result = launch({"PREFILL_COMPUTE_SHARE": "auto", name: value})
    assert result.returncode == 2
    assert name in result.stderr


def test_numeric_share_with_half_life_fails(launch):
    assert (
        launch(
            {"PREFILL_COMPUTE_SHARE": "0.4", "PREFILL_COMPUTE_HALF_LIFE": "smooth"}
        ).returncode
        == 2
    )


def test_cli_effective_share_controls_half_life_validation(launch):
    assert (
        launch(
            {"PREFILL_COMPUTE_SHARE": "auto", "PREFILL_COMPUTE_HALF_LIFE": "smooth"},
            ["--prefill-compute-share", "0.4"],
        ).returncode
        == 2
    )


def test_compatibility_selector_and_explicit_cli(launch):
    environment = {"FAIRNESS_ENGINE": "none", "PREFILL_COMPUTE_SHARE": "0.4"}
    assert argv(launch(environment)) == []
    assert argv(launch(environment, ["--prefill-compute-share", "auto"])) == [
        "--prefill-compute-share",
        "auto",
    ]
    assert launch({"FAIRNESS_ENGINE": "compute_share"}).returncode == 2
    assert launch({"FAIRNESS_ENGINE": "micro_slicing"}).returncode == 2


@pytest.mark.parametrize(
    "option", list(OPTIONS.values()) + ["--prefill-schedule-interval"]
)
def test_duplicate_and_missing_cli_values_fail(launch, option):
    assert launch(arguments=[option, "auto", f"{option}=auto"]).returncode == 2
    assert launch(arguments=[option]).returncode == 2


def test_fairness_interval_uses_cli_precedence(launch):
    environment = {
        "PREFILL_COMPUTE_SHARE": "0.4",
        "PREFILL_SCHEDULE_INTERVAL": "invalid",
    }
    arguments = ["--prefill-schedule-interval=1"]
    assert argv(launch(environment, arguments)) == arguments + [
        "--prefill-compute-share",
        "0.4",
    ]
    assert launch(environment).returncode == 2
    assert (
        launch(
            {"PREFILL_COMPUTE_SHARE": "0.4"}, ["--prefill-schedule-interval", "8"]
        ).returncode
        == 2
    )


def test_base_interval_is_not_duplicated_and_chat_defaults_are_explicit(launch):
    result = launch(
        {"PREFILL_SCHEDULE_INTERVAL": "invalid"},
        ["--prefill-schedule-interval", "1"],
        base=True,
    )
    assert result.returncode == 0, result.stderr
    arguments = shlex.split(result.stdout.split("launch:", 1)[1])
    assert arguments.count("--prefill-schedule-interval") == 1
    assert json.loads(
        arguments[arguments.index("--default-chat-template-kwargs") + 1]
    ) == {"reasoning_effort": "high", "clear_thinking": False}


@pytest.mark.parametrize("equals", [False, True])
def test_explicit_chat_defaults_are_forwarded_without_a_second_object(launch, equals):
    value = '{"reasoning_effort":"max","clear_thinking":true}'
    option = "--default-chat-template-kwargs"
    supplied = [f"{option}={value}"] if equals else [option, value]
    result = launch(arguments=supplied, base=True)
    assert result.returncode == 0, result.stderr
    arguments = shlex.split(result.stdout.split("launch:", 1)[1])
    settings = [item for item in arguments if item.split("=", 1)[0] == option]
    assert len(settings) == 1
    if equals:
        assert settings == [f"{option}={value}"]
    else:
        assert arguments[arguments.index(option) + 1] == value


def test_graph_capture_controls_are_preserved(launch):
    assert argv(launch({"CUDAGRAPH_CAPTURE_SIZES": "1 4 8"})) == [
        "--cudagraph-capture-sizes",
        "1",
        "4",
        "8",
    ]


def test_help_describes_all_scheduler_controls(launch):
    result = launch(arguments=["--help"])
    assert result.returncode == 0
    for environment, option in OPTIONS.items():
        assert environment in result.stdout
        assert option in result.stdout
