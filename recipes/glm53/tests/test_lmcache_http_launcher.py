"""Validate sidecar bind arguments without starting a cache or model server."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest

RECIPE = Path(__file__).resolve().parents[1]
WRAPPERS = (
    "serve-glm53-flash-lmcache.sh",
    "serve-glm53-flash-lmcache-cache-complete.sh",
)


def render(wrapper, host=None, dtype=None):
    source = (RECIPE / wrapper).read_text()
    # The configuration section validates settings and constructs argv. Stop
    # before child-process supervision; no sidecar or vLLM process is launched.
    configuration, separator, _ = source.partition("lmcache_pid=\n")
    assert separator
    if dtype is not None:
        # The standalone wrapper resolves vLLM's dtype after defining its
        # supervisor functions, but before broker or child-process creation.
        configuration, separator, _ = source.partition(
            "if [[ ${transfer_mode} != engine_driven ]]; then"
        )
        assert separator
    environment = {
        "PATH": os.environ["PATH"],
        "LMCACHE_ENABLED": "1",
        "LMCACHE_L2_ENABLED": "0",
        "LMCACHE_MIN_SHM_GIB": "1",
    }
    if host is not None:
        environment["LMCACHE_HTTP_HOST"] = host
    if dtype is not None:
        environment["LMCACHE_KV_CACHE_DTYPE"] = dtype
    return subprocess.run(
        [
            "bash",
            "-c",
            configuration
            + '\nprintf "%s\\0" "${health_url}" "${lmcache_server[@]}" "vllm_dtype=${KV_CACHE_DTYPE:-}"',
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize(
    "host,expected,probe,warning",
    [
        (None, "127.0.0.1", "127.0.0.1", False),
        ("0.0.0.0", "0.0.0.0", "127.0.0.1", True),
        ("192.0.2.10", "192.0.2.10", "192.0.2.10", True),
        ("::", "::", "[::1]", True),
        ("::1", "::1", "[::1]", False),
        ("cache.internal", "cache.internal", "cache.internal", True),
        ("127.example.internal", "127.example.internal", "127.example.internal", True),
        ("127.0.0.2", "127.0.0.2", "127.0.0.2", False),
        ("127.999.0.1", "127.999.0.1", "127.999.0.1", True),
    ],
)
def test_http_bind_and_readiness_use_compatible_addresses(
    wrapper, host, expected, probe, warning
):
    result = render(wrapper, host)
    assert result.returncode == 0, result.stderr
    fields = result.stdout.rstrip("\0").split("\0")
    assert fields[0] == f"http://{probe}:8085/healthcheck"
    assert fields.count("--http-host") == 1
    assert fields[fields.index("--http-host") + 1] == expected
    assert ("administrative APIs" in result.stderr) == warning


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize(
    "host", ["--host", "localhost;id", "a b", "$(id)", "a/b", "a\nb"]
)
def test_invalid_bind_is_rejected_before_process_start(wrapper, host):
    result = render(wrapper, host)
    assert result.returncode == 2
    assert "LMCACHE_HTTP_HOST" in result.stderr


@pytest.mark.parametrize(
    "dtype,expected",
    [
        ("fp8_ds_mla", "fp8"),
        ("fp8", "fp8"),
        ("fp8_e4m3", "fp8_e4m3"),
        ("nvfp4_ds_mla", "nvfp4_ds_mla"),
    ],
)
def test_standalone_cache_storage_dtype_maps_to_vllm(dtype, expected):
    result = render("serve-glm53-flash-lmcache.sh", dtype=dtype)
    assert result.returncode == 0, result.stderr
    assert result.stdout.rstrip("\0").split("\0")[-1] == f"vllm_dtype={expected}"


@pytest.mark.parametrize(
    "transfer,resolved",
    [("engine_driven", "request_boundaries"), ("lmcache_driven", "aligned")],
)
@pytest.mark.parametrize(
    "policy_args",
    [
        [],
        ["--recurrent-checkpoint-policy", "auto"],
        ["--recurrent-checkpoint-policy=auto"],
    ],
)
def test_cache_policy_is_resolved_once_before_delegation(
    transfer, resolved, policy_args
):
    unrelated = ["--max-model-len", "65536", "--served-model-name", "model with spaces"]
    result = subprocess.run(
        [
            "bash",
            str(RECIPE / "serve-glm53-flash-cache-complete.sh"),
            *policy_args,
            *unrelated,
        ],
        env={
            "PATH": os.environ["PATH"],
            "CACHE_CONFIG_DRY_RUN": "1",
            "CACHE_MODE": "lmcache",
            "LMCACHE_TRANSFER_MODE": transfer,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    line = next(line for line in result.stdout.splitlines() if line.startswith("ARGV:"))
    args = shlex.split(line.removeprefix("ARGV:"))
    assert args == unrelated + ["--recurrent-checkpoint-policy", resolved]


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--recurrent-checkpoint-policy", "request_boundaries"],
        ["--recurrent-checkpoint-policy=request_boundaries"],
    ],
)
def test_semantic_sidecar_does_not_duplicate_native_checkpoint_policy(args):
    source = (RECIPE / "serve-glm53-flash-lmcache-cache-complete.sh").read_text()
    configuration, separator, _ = source.partition("lmcache_pid=\n")
    assert separator
    _, separator, connector = source.partition("connector_config=$(printf")
    assert separator
    connector, separator, _ = connector.partition('"${base_launcher}" "$@"')
    assert separator
    script = (
        configuration
        + "\nvllm_extra_args=()\nconnector_config=$(printf"
        + connector
        + '\nprintf "%s\\0" "$@" "${vllm_extra_args[@]}"'
    )
    result = subprocess.run(
        ["bash", "-c", script, "policy-test", *args],
        env={
            "PATH": os.environ["PATH"],
            "LMCACHE_ENABLED": "1",
            "LMCACHE_L2_ENABLED": "0",
            "LMCACHE_MIN_SHM_GIB": "1",
            "LMCACHE_TRANSFER_MODE": "engine_driven",
            "LMCACHE_CHECKPOINT_IDENTITY": '{"target_revision":"test"}',
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    fields = result.stdout.rstrip("\0").split("\0")
    assert (
        sum(
            field.split("=", 1)[0] == "--recurrent-checkpoint-policy"
            for field in fields
        )
        == 1
    )
