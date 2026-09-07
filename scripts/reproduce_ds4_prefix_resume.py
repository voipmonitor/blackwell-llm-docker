#!/usr/bin/env python3
"""Stress GPU memory with shared-prefix resumes and concurrent decode.

The inputs are synthetic token IDs, not a model-quality evaluation. Run the
client on the serving host so --gpu-ids samples the server's devices. A completed
run qualifies only its recorded workload, not an arbitrary serving duration.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPException
from pathlib import Path
from urllib.request import urlopen

from qualify_ds4_memory_admission import (
    _read_free_memory,
    _request_json,
    _resolve_model,
    _sample_memory,
)


def request_payload(
    model, prefix_tokens, tail_tokens, tail_marker, output_tokens, prefix_marker=0
):
    # ASCII A-J encode decimal markers; K terminates each marker so repeated
    # digit strings such as 1 and 11 cannot create identical cached tails.
    marker = [35 + int(digit) for digit in str(tail_marker)] + [45]
    tail = (marker * ((tail_tokens + len(marker) - 1) // len(marker)))[:tail_tokens]
    prefix = [35] * prefix_tokens
    if prefix_marker:
        identity = [35 + int(digit) for digit in str(prefix_marker)] + [45]
        prefix[: min(len(identity), prefix_tokens)] = identity[:prefix_tokens]
    return {
        "model": model,
        "prompt": prefix + tail,
        "max_tokens": output_tokens,
        "temperature": 1.0,
        "seed": 0,
        "ignore_eos": True,
        "stream": False,
    }


def submit(base_url, name, payload, expected_cached_tokens, timeout, barrier=None):
    started = time.monotonic()
    result = {"name": name, "error": None, "http_status": None}
    try:
        if barrier is not None:
            barrier.wait(timeout=min(timeout, 30))
        started = time.monotonic()
        status, response = _request_json(
            "POST", f"{base_url}/v1/completions", payload, timeout
        )
        usage = response.get("usage") or {}
        cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        result.update(
            http_status=status,
            usage=usage,
            prompt_tokens_match=usage.get("prompt_tokens") == len(payload["prompt"]),
            output_tokens_match=usage.get("completion_tokens") == payload["max_tokens"],
            expected_cached_tokens=expected_cached_tokens,
            prefix_hit_matches=(
                None
                if expected_cached_tokens is None
                else cached_tokens == expected_cached_tokens
            ),
            finish_reasons=[
                choice.get("finish_reason") for choice in response.get("choices", [])
            ],
        )
    except (OSError, HTTPException, ValueError, RuntimeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def outcome(results, health_status):
    if health_status != 200 or any(
        result["http_status"] != 200 or result["error"] for result in results
    ):
        return "failed"
    if any(
        not result["prompt_tokens_match"]
        or not result["output_tokens_match"]
        or result["prefix_hit_matches"] is False
        for result in results
    ):
        return "inconclusive"
    return "completed"


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_float(value):
    parsed = float(value)
    if not 0 <= parsed < float("inf"):
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--prefix-tokens", type=positive_int, default=267008)
    parser.add_argument("--expected-cached-tokens", type=positive_int)
    parser.add_argument("--resume-tokens", type=positive_int, default=3723)
    parser.add_argument("--concurrency", type=positive_int, default=4)
    parser.add_argument("--waves", type=positive_int, default=4)
    parser.add_argument("--output-tokens", type=positive_int, default=4096)
    parser.add_argument("--independent-prefixes", action="store_true")
    parser.add_argument("--stagger-seconds", type=nonnegative_float, default=0)
    parser.add_argument("--timeout", type=positive_int, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected_cached = args.expected_cached_tokens or args.prefix_tokens
    base_url = args.base_url.rstrip("/")
    model = _resolve_model(base_url, args.model, args.timeout)
    samples = [_read_free_memory(args.gpu_ids)]
    stop = threading.Event()
    sampler = threading.Thread(
        target=_sample_memory, args=(args.gpu_ids, stop, samples, 0.25), daemon=True
    )
    results = []
    sampler.start()
    try:
        # The extra token permits complete prefix blocks to be cached; DSpark
        # may still back off by a block according to its target-state contract.
        prefix_count = args.concurrency if args.independent_prefixes else 1
        for request in range(prefix_count):
            priming = request_payload(
                model,
                args.prefix_tokens,
                1,
                0,
                1,
                prefix_marker=request + 1 if args.independent_prefixes else 0,
            )
            result = submit(
                base_url, f"prefix-prime-{request}", priming, None, args.timeout
            )
            results.append(result)
            if result["http_status"] != 200 or result["error"]:
                break
        if all(
            result["http_status"] == 200 and not result["error"] for result in results
        ):
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                for wave in range(args.waves):
                    barrier = (
                        None
                        if args.stagger_seconds
                        else threading.Barrier(args.concurrency)
                    )
                    futures = []
                    for request in range(args.concurrency):
                        # Resume markers differ within the first cache block;
                        # a prior wave therefore cannot supply a cached tail.
                        payload = request_payload(
                            model,
                            args.prefix_tokens,
                            args.resume_tokens,
                            1 + wave * args.concurrency + request,
                            args.output_tokens,
                            prefix_marker=(
                                request + 1 if args.independent_prefixes else 0
                            ),
                        )
                        futures.append(
                            executor.submit(
                                submit,
                                base_url,
                                f"resume-{wave}-{request}",
                                payload,
                                expected_cached,
                                args.timeout,
                                barrier,
                            )
                        )
                        if args.stagger_seconds and request + 1 < args.concurrency:
                            time.sleep(args.stagger_seconds)
                    wave_results = [future.result() for future in futures]
                    results.extend(wave_results)
                    if any(result["error"] for result in wave_results):
                        break
    finally:
        stop.set()
        sampler.join()

    health_status = None
    health_error = None
    try:
        with urlopen(f"{base_url}/health", timeout=15) as response:
            health_status = response.status
    except (OSError, HTTPException, ValueError, RuntimeError) as exc:
        health_error = str(exc)

    receipt = {
        "schema_version": 1,
        "status": "research-only",
        "outcome": outcome(results, health_status),
        "purpose": "shared-prefix resume and concurrent decode memory stress",
        "configuration": {**vars(args), "output": str(args.output), "model": model},
        "results": results,
        "minimum_free_mib": [min(values) for values in zip(*samples, strict=True)],
        "memory_samples": len(samples),
        "health_status": health_status,
        "health_error": health_error,
        "limitations": [
            "Synthetic token inputs do not reproduce the reporter's exact request history.",
            "Host memory sampling can miss allocations shorter than its 250 ms interval.",
            "HTTP overlap does not prove a particular mixed scheduler batch was executed.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["outcome"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
