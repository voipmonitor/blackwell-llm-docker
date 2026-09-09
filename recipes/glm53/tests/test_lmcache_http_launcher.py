"""Validate sidecar bind arguments without starting a cache or model server."""

import os
import subprocess
from pathlib import Path

import pytest

RECIPE = Path(__file__).resolve().parents[1]
WRAPPERS = (
    "serve-glm53-flash-lmcache.sh",
    "serve-glm53-flash-lmcache-cache-complete.sh",
)


def render(wrapper, host=None):
    source = (RECIPE / wrapper).read_text()
    # The configuration section validates settings and constructs argv. Stop
    # before child-process supervision; no sidecar or vLLM process is launched.
    configuration, separator, _ = source.partition("lmcache_pid=\n")
    assert separator
    environment = {
        "PATH": os.environ["PATH"],
        "LMCACHE_ENABLED": "1",
        "LMCACHE_L2_ENABLED": "0",
        "LMCACHE_MIN_SHM_GIB": "1",
    }
    if host is not None:
        environment["LMCACHE_HTTP_HOST"] = host
    return subprocess.run(
        ["bash", "-c", configuration + '\nprintf "%s\\0" "${health_url}" "${lmcache_server[@]}"'],
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
    ],
)
def test_http_bind_and_readiness_use_compatible_addresses(wrapper, host, expected, probe, warning):
    result = render(wrapper, host)
    assert result.returncode == 0, result.stderr
    fields = result.stdout.rstrip("\0").split("\0")
    assert fields[0] == f"http://{probe}:8085/healthcheck"
    assert fields.count("--http-host") == 1
    assert fields[fields.index("--http-host") + 1] == expected
    assert ("administrative APIs" in result.stderr) == warning


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize("host", ["--host", "localhost;id", "a b", "$(id)", "a/b", "a\nb"])
def test_invalid_bind_is_rejected_before_process_start(wrapper, host):
    result = render(wrapper, host)
    assert result.returncode == 2
    assert "LMCACHE_HTTP_HOST" in result.stderr
