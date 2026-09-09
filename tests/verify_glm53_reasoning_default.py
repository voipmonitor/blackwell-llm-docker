"""Verify GLM reasoning rendering with installed vLLM and real tokenizer files.

Run inside a GLM serving image with CUDA_VISIBLE_DEVICES empty. This checks
request/default precedence and the exact model-visible prompt; it does not
infer reasoning behavior from output length.
"""

import argparse
import json

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tokenizers import get_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    tokenizer = get_tokenizer(args.model)
    rows = []
    for label, defaults, extra, expected in (
        ("checkpoint_template_default", {}, {}, "Max"),
        ("server_default", {"reasoning_effort": "high"}, {}, "High"),
        ("explicit_max", {"reasoning_effort": "high"}, {"reasoning_effort": "max"}, "Max"),
        ("explicit_high", {"reasoning_effort": "high"}, {"reasoning_effort": "high"}, "High"),
        ("explicit_low", {"reasoning_effort": "high"}, {"reasoning_effort": "low"}, "Low"),
        ("template_max", {"reasoning_effort": "high"},
         {"chat_template_kwargs": {"reasoning_effort": "max"}}, "Max"),
    ):
        request = ChatCompletionRequest(
            model="GLM-5.3-Flash-NVFP4",
            messages=[{"role": "user", "content": "What is 2 + 2?"}],
            **extra,
        )
        params = request.build_chat_params(None, "auto").with_defaults(defaults)
        rendered = tokenizer.apply_chat_template(
            request.messages, tokenize=False, **params.get_apply_chat_template_kwargs()
        )
        assert rendered.startswith(f"[gMASK]<sop><|system|>Reasoning Effort: {expected}<|user|>"), label
        rows.append({"case": label, "effort": expected.lower(), "prompt": rendered})
    print(json.dumps({"status": "passed", "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
