#!/usr/bin/env python3
"""Exercise scheduler-reachable DeepSeek V4 memory peaks through the API.

The server must already be running. The client submits an exact token-ID
prefill equal to ``max_num_batched_tokens`` and can overlap a long text prefill
with multiple image encodes. Host-side GPU sampling records the minimum free
memory observed while the requests execute.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import struct
import subprocess
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zlib


@dataclass
class RequestResult:
    name: str
    status: int | None
    elapsed_seconds: float
    usage: dict[str, Any] | None
    expected_prompt_tokens: int | None
    prompt_token_contract_satisfied: bool | None
    error: str | None


def _request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    timeout: float,
) -> tuple[int, dict[str, Any]]:
    body = None if payload is None else json.dumps(payload).encode()
    request = Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        response_body = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {response_body[:4096]}") from exc
    except URLError as exc:
        raise RuntimeError(f"request failed: {exc.reason}") from exc


def _resolve_model(base_url: str, requested: str | None, timeout: float) -> str:
    if requested:
        return requested
    status, payload = _request_json("GET", f"{base_url}/v1/models", None, timeout)
    if status != 200 or not payload.get("data"):
        raise RuntimeError("/v1/models did not return a served model")
    return str(payload["data"][0]["id"])


def _run_completion(
    base_url: str,
    model: str,
    name: str,
    prompt_tokens: int,
    token_id: int,
    timeout: float,
) -> RequestResult:
    started = time.monotonic()
    try:
        status, payload = _request_json(
            "POST",
            f"{base_url}/v1/completions",
            {
                "model": model,
                "prompt": [token_id] * prompt_tokens,
                "max_tokens": 1,
                "temperature": 0,
                "stream": False,
            },
            timeout,
        )
        return RequestResult(
            name=name,
            status=status,
            elapsed_seconds=time.monotonic() - started,
            usage=payload.get("usage"),
            expected_prompt_tokens=prompt_tokens,
            prompt_token_contract_satisfied=(
                payload.get("usage", {}).get("prompt_tokens") == prompt_tokens
            ),
            error=None,
        )
    except Exception as exc:
        return RequestResult(
            name=name,
            status=None,
            elapsed_seconds=time.monotonic() - started,
            usage=None,
            expected_prompt_tokens=prompt_tokens,
            prompt_token_contract_satisfied=False,
            error=str(exc),
        )


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    crc = binascii.crc32(kind)
    crc = binascii.crc32(data, crc)
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def _solid_png_data_url(size: int, rgb: tuple[int, int, int]) -> str:
    row = b"\x00" + bytes(rgb) * size
    raw = row * size
    png = b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(raw, level=1)),
            _png_chunk(b"IEND", b""),
        )
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _run_vision(
    base_url: str,
    model: str,
    image_count: int,
    image_size: int,
    timeout: float,
) -> RequestResult:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "Describe the dominant color in each image."}
    ]
    for index in range(image_count):
        color = (
            (37 * index + 31) % 256,
            (73 * index + 67) % 256,
            (109 * index + 101) % 256,
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _solid_png_data_url(image_size, color)},
            }
        )

    started = time.monotonic()
    try:
        status, payload = _request_json(
            "POST",
            f"{base_url}/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 1,
                "temperature": 0,
                "stream": False,
            },
            timeout,
        )
        return RequestResult(
            name=f"vision-{image_count}x{image_size}",
            status=status,
            elapsed_seconds=time.monotonic() - started,
            usage=payload.get("usage"),
            expected_prompt_tokens=None,
            prompt_token_contract_satisfied=None,
            error=None,
        )
    except Exception as exc:
        return RequestResult(
            name=f"vision-{image_count}x{image_size}",
            status=None,
            elapsed_seconds=time.monotonic() - started,
            usage=None,
            expected_prompt_tokens=None,
            prompt_token_contract_satisfied=None,
            error=str(exc),
        )


def _read_free_memory(gpu_ids: str) -> list[int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_ids}",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [int(line.strip()) for line in completed.stdout.splitlines() if line.strip()]


def _sample_memory(
    gpu_ids: str,
    stop: threading.Event,
    samples: list[list[int]],
    interval: float,
) -> None:
    while not stop.wait(interval):
        try:
            samples.append(_read_free_memory(gpu_ids))
        except (OSError, subprocess.SubprocessError, ValueError):
            continue


def _is_success(result: RequestResult) -> bool:
    return (
        result.status == 200
        and result.error is None
        and result.prompt_token_contract_satisfied is not False
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--prefill-tokens", type=int, default=4096)
    parser.add_argument("--long-prefill-tokens", type=int, default=810000)
    parser.add_argument("--token-id", type=int, default=100)
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--image-count", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=2048)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    model = _resolve_model(base_url, args.model, args.timeout)
    results = [
        _run_completion(
            base_url,
            model,
            "full-scheduler-quantum",
            args.prefill_tokens,
            args.token_id,
            args.timeout,
        )
    ]

    samples = [_read_free_memory(args.gpu_ids)]
    if args.long_prefill_tokens > 0:
        stop = threading.Event()
        sampler = threading.Thread(
            target=_sample_memory,
            args=(args.gpu_ids, stop, samples, args.sample_interval),
            daemon=True,
        )
        sampler.start()
        try:
            with ThreadPoolExecutor(max_workers=2 if args.vision else 1) as executor:
                futures = [
                    executor.submit(
                        _run_completion,
                        base_url,
                        model,
                        "long-prefill",
                        args.long_prefill_tokens,
                        args.token_id + 1,
                        args.timeout,
                    )
                ]
                if args.vision:
                    futures.append(
                        executor.submit(
                            _run_vision,
                            base_url,
                            model,
                            args.image_count,
                            args.image_size,
                            args.timeout,
                        )
                    )
                results.extend(future.result() for future in futures)
        finally:
            stop.set()
            sampler.join()
            samples.append(_read_free_memory(args.gpu_ids))

    health_status: int | None = None
    health_error: str | None = None
    try:
        health_status, _ = _request_json(
            "GET", f"{base_url}/v1/models", None, args.timeout
        )
    except Exception as exc:
        health_error = str(exc)
    minimum_free_mib = [min(values) for values in zip(*samples, strict=True)]
    status = (
        "qualified"
        if health_status == 200 and all(_is_success(result) for result in results)
        else "failed"
    )
    receipt = {
        "schema_version": 1,
        "status": status,
        "purpose": (
            "DeepSeek V4 GPU KV admission qualification at the complete "
            "scheduler token quantum"
        ),
        "server": {"base_url": base_url, "model": model},
        "configuration": {
            "gpu_ids": [int(value) for value in args.gpu_ids.split(",")],
            "prefill_tokens": args.prefill_tokens,
            "long_prefill_tokens": args.long_prefill_tokens,
            "vision": args.vision,
            "image_count": args.image_count if args.vision else 0,
            "image_size": args.image_size if args.vision else 0,
        },
        "results": [asdict(result) for result in results],
        "minimum_free_mib": minimum_free_mib,
        "memory_samples": len(samples),
        "service_healthy_after": health_status == 200,
        "service_health_error": health_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if status == "qualified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
