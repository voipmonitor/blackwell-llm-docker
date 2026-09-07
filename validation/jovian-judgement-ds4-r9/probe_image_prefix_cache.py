"""Measure GPU token-cache reuse for identical short and long image prompts.

The request places a JPEG before repeated reference notes. Inspect prompt
and cached-token counts separately from answer content: the 128-token output
budget can end during reasoning. This is not a task-quality benchmark.
"""

import argparse
import base64
import hashlib
import json
import time
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    image_bytes = args.image.read_bytes()
    image_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
    records = []
    for repetitions in (0, 320):
        text = (
            "Read the reference notes and identify the pictured ingredient.\n"
            + "Reference note: sample 47 belongs to the botanical collection.\n"
            * repetitions
        )
        payload = {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": text},
                        {
                            "type": "text",
                            "text": "Name the ingredient in one sentence.",
                        },
                    ],
                }
            ],
            "temperature": 1.0,
            "max_tokens": 128,
        }
        for repeat in range(3):
            started = time.monotonic()
            response = requests.post(
                args.base + "/v1/chat/completions", json=payload, timeout=900
            )
            response.raise_for_status()
            body = response.json()
            record = {
                "reference_repetitions": repetitions,
                "repeat": repeat,
                "elapsed_seconds": time.monotonic() - started,
                "usage": body["usage"],
                "message": body["choices"][0]["message"],
                "finish_reason": body["choices"][0]["finish_reason"],
            }
            records.append(record)
            print(json.dumps(record), flush=True)
            args.output.write_text(
                json.dumps(
                    {
                        "model": args.model,
                        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                        "temperature": 1.0,
                        "image_position": "before_reference_notes",
                        "records": records,
                    },
                    indent=2,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
