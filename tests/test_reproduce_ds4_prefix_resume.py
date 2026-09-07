import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from reproduce_ds4_prefix_resume import (
    nonnegative_float,
    outcome,
    request_payload,
    submit,
)


def test_resume_payloads_diverge_after_the_complete_prefix():
    prime = request_payload("model", 267008, 1, 0, 1)
    left = request_payload("model", 267008, 3723, 1, 4096)
    right = request_payload("model", 267008, 3723, 2, 4096)
    assert len(left["prompt"]) == len(right["prompt"]) == 270731
    assert (
        prime["prompt"][:267008] == left["prompt"][:267008] == right["prompt"][:267008]
    )
    assert prime["prompt"][267008] != left["prompt"][267008] != right["prompt"][267008]
    assert left["ignore_eos"] is True
    assert left["max_tokens"] == 4096


def test_large_resume_markers_use_ascii_token_ids_and_distinct_tails():
    left = request_payload("model", 8, 3723, 1000, 16)
    right = request_payload("model", 8, 3723, 1001, 16)
    assert len(left["prompt"]) == 3731
    assert set(left["prompt"]) <= set(range(35, 46))
    assert left["prompt"][8:264] != right["prompt"][8:264]


def test_repeated_digit_markers_do_not_share_a_cached_continuation():
    tails = [
        tuple(request_payload("model", 8, 4096, marker, 16)["prompt"][8:264])
        for marker in range(1000)
    ]
    assert len(set(tails)) == len(tails)


def test_independent_prefixes_preserve_each_agents_resume_identity():
    prime = request_payload("model", 1024, 1, 0, 1, prefix_marker=1)
    resume = request_payload("model", 1024, 512, 1, 16, prefix_marker=1)
    separate = request_payload("model", 1024, 512, 1, 16, prefix_marker=2)
    assert prime["prompt"][:1024] == resume["prompt"][:1024]
    assert resume["prompt"][:256] != separate["prompt"][:256]
    assert len(resume["prompt"]) == len(separate["prompt"]) == 1536


@pytest.mark.parametrize("value", ["nan", "inf", "-1"])
def test_stagger_interval_rejects_unbounded_or_negative_values(value):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        nonnegative_float(value)


@pytest.mark.parametrize(
    "cached_tokens,expected_outcome",
    [(267008, "completed"), (0, "inconclusive"), (None, "inconclusive")],
)
def test_success_requires_the_recorded_prefix_hit(
    monkeypatch, cached_tokens, expected_outcome
):
    def request(*args):
        return 200, {
            "usage": {
                "prompt_tokens": 270731,
                "completion_tokens": 4096,
                "prompt_tokens_details": {"cached_tokens": cached_tokens},
            }
        }

    monkeypatch.setattr("reproduce_ds4_prefix_resume._request_json", request)
    result = submit(
        "http://localhost",
        "resume",
        request_payload("model", 267008, 3723, 102, 4096),
        267008,
        10,
    )
    assert outcome([result], 200) == expected_outcome
    assert outcome([result], 503) == "failed"


def test_request_failure_remains_an_explicit_failed_result(monkeypatch):
    def request(*args):
        raise RuntimeError("HTTP 500: engine unavailable")

    monkeypatch.setattr("reproduce_ds4_prefix_resume._request_json", request)
    result = submit(
        "http://localhost", "resume", request_payload("model", 8, 3, 102, 1), 8, 10
    )
    assert "engine unavailable" in result["error"]
    assert outcome([result], 200) == "failed"
