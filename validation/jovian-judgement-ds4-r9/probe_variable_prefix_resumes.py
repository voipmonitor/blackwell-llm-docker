"""Measure allocation headroom across 267k-token cached continuations.

Prime a 267,264-token prefix, then submit deterministic waves of four requests
with 257-8,192 appended tokens and 256 output tokens. Every continuation must
report exactly 267,008 cached tokens. This measures a bounded serving sequence,
not task-level output quality or equivalence to a production request history.
"""

import argparse
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from reproduce_ds4_prefix_resume import (
    _read_free_memory,
    _sample_memory,
    positive_int,
    request_payload,
    submit,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--waves", type=positive_int, default=24)
    args = parser.parse_args()
    rng = random.Random(731)
    results = []
    samples = [_read_free_memory(args.gpu_ids)]
    stop = threading.Event()
    sampler = threading.Thread(
        target=_sample_memory, args=(args.gpu_ids, stop, samples, 0.25)
    )
    sampler.start()
    try:
        prime = submit(
            args.base_url,
            "prime",
            request_payload(args.model, 267264, 1, 0, 1, prefix_marker=731),
            None,
            1800,
        )
        results.append(prime)
        if prime["http_status"] != 200 or prime["error"]:
            raise RuntimeError(f"Prefix priming failed: {prime}")
        with ThreadPoolExecutor(max_workers=4) as pool:
            for wave in range(args.waves):
                barrier = threading.Barrier(4)
                futures = []
                for slot in range(4):
                    tail = rng.randint(257, 8192)
                    payload = request_payload(
                        args.model,
                        267264,
                        tail,
                        10000 + wave * 4 + slot,
                        256,
                        prefix_marker=731,
                    )
                    futures.append(
                        pool.submit(
                            submit,
                            args.base_url,
                            f"wave-{wave}-slot-{slot}-tail-{tail}",
                            payload,
                            267008,
                            1800,
                            barrier,
                        )
                    )
                batch = [future.result() for future in futures]
                results.extend(batch)
                receipt = {
                    "model": args.model,
                    "seed": 731,
                    "results": results,
                    "minimum_free_mib": [min(values) for values in zip(*samples)],
                    "free_mib_after_wave": _read_free_memory(args.gpu_ids),
                }
                args.output.write_text(json.dumps(receipt, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "wave": wave,
                            "ok": sum(result["http_status"] == 200 for result in batch),
                            "cache_hits_match": all(
                                result.get("prefix_hit_matches") for result in batch
                            ),
                            "minimum_free_mib": receipt["minimum_free_mib"],
                            "free_mib_after_wave": receipt["free_mib_after_wave"],
                        }
                    ),
                    flush=True,
                )
                if any(
                    result["http_status"] != 200
                    or result["error"]
                    or not result["prefix_hit_matches"]
                    for result in batch
                ):
                    raise RuntimeError(
                        "A continuation failed its HTTP or prefix contract"
                    )
    finally:
        stop.set()
        sampler.join()


if __name__ == "__main__":
    main()
