#!/usr/bin/env python3
"""KLD evaluation for quantized models via SGLang.

Measures KL divergence between a reference model (FP8) and test models
(e.g. NVFP4) using full log-probability distributions captured by the
sglang-kld-logit-capture patch.

Three phases:
  --phase ref     Send sliding windows to server, logits saved by patch
  --phase test    Same, for test model
  --phase compute Load ref + test logits, compute KLD, print results

Example workflow:
  # Phase 1: Start FP8 ref server with patch, then:
  python scripts/sglang_kld_eval.py --phase ref \\
    --server-url http://localhost:5000 \\
    --tokenizer Qwen/Qwen3.5-397B-A17B-FP8 \\
    --logits-dir /tmp/kld_ref

  # Phase 2: Start NVFP4 test server with patch, then:
  python scripts/sglang_kld_eval.py --phase test \\
    --server-url http://localhost:5000 \\
    --tokenizer Qwen/Qwen3.5-397B-A17B-FP8 \\
    --logits-dir /tmp/kld_test

  # Phase 3: Compute KLD
  python scripts/sglang_kld_eval.py --phase compute \\
    --ref-dir /tmp/kld_ref \\
    --test-dirs /tmp/kld_test \\
    --test-names "NVFP4"
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import requests
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer


def load_wikitext(config="wikitext-2-raw-v1"):
    """Load and concatenate wikitext dataset."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", config, split="test")
    # Concatenate all text with newlines (matching standard PPL eval)
    return "\n\n".join(ds["text"])


def build_sliding_windows(token_ids, context_length=2048, stride=512, num_windows=100):
    """Build sliding windows over tokenized text.

    Returns list of token ID lists, each of length context_length.
    """
    total_needed = context_length + (num_windows - 1) * stride
    if len(token_ids) < total_needed:
        print(
            f"Warning: only {len(token_ids)} tokens available, "
            f"need {total_needed}. Reducing num_windows."
        )
        num_windows = (len(token_ids) - context_length) // stride + 1
        if num_windows < 1:
            raise ValueError(
                f"Not enough tokens ({len(token_ids)}) for even one "
                f"window of {context_length}"
            )

    windows = []
    for i in range(num_windows):
        start = i * stride
        end = start + context_length
        windows.append(token_ids[start:end])
    return windows


def send_prefill_request(server_url, token_ids, timeout=300):
    """Send a prefill-only request to SGLang to trigger logit capture.

    Uses the /generate endpoint with max_new_tokens=1 and return_logprob
    to force input logprob computation (which triggers our capture hook).
    """
    payload = {
        "input_ids": token_ids,
        "sampling_params": {
            "max_new_tokens": 1,
            "temperature": 0.0,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    resp = requests.post(
        f"{server_url}/generate",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def wait_for_server(server_url, timeout=600):
    """Wait for SGLang server to be ready."""
    print(f"Waiting for server at {server_url} ...", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{server_url}/health", timeout=5)
            if r.status_code == 200:
                print(" ready.")
                return
        except requests.ConnectionError:
            pass
        print(".", end="", flush=True)
        time.sleep(5)
    raise TimeoutError(f"Server not ready after {timeout}s")


def run_logit_generation(args):
    """Phase ref/test: send sliding windows to server for logit capture."""
    print(f"=== Phase: {args.phase} ===")
    print(f"Server: {args.server_url}")
    print(f"Logits dir: {args.logits_dir}")

    wait_for_server(args.server_url)

    # Check logits dir
    logits_dir = Path(args.logits_dir)
    existing = list(logits_dir.glob("*.safetensors"))
    if existing:
        print(
            f"Warning: {len(existing)} safetensors files already in {logits_dir}. "
            f"These will be mixed with new files. Consider clearing the directory."
        )

    # Load and tokenize dataset
    print("Loading wikitext dataset...")
    text = load_wikitext(args.dataset_config)
    print(f"Loaded {len(text)} characters")

    print(f"Tokenizing with {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    token_ids = tokenizer.encode(text)
    print(f"Tokenized to {len(token_ids)} tokens")

    # Build sliding windows
    windows = build_sliding_windows(
        token_ids,
        context_length=args.context_length,
        stride=args.stride,
        num_windows=args.num_windows,
    )
    print(
        f"Built {len(windows)} sliding windows "
        f"(context={args.context_length}, stride={args.stride})"
    )
    total_positions = len(windows) * (args.context_length - 1)
    est_storage_gb = (
        total_positions * args.vocab_size * 2 / (1024**3)
    )
    print(
        f"Target positions: {total_positions:,} "
        f"(estimated storage: {est_storage_gb:.1f} GB)"
    )

    # Send each window to the server
    t0 = time.time()
    for i, window in enumerate(windows):
        t_start = time.time()
        try:
            send_prefill_request(args.server_url, window, timeout=args.timeout)
        except Exception as e:
            print(f"\nError on window {i}: {e}")
            raise
        elapsed = time.time() - t_start

        # Verify file was saved
        expected_file = logits_dir / f"{i}.safetensors"
        if expected_file.exists():
            size_mb = expected_file.stat().st_size / (1024**2)
            print(
                f"  Window {i+1}/{len(windows)}: {elapsed:.1f}s, "
                f"saved {size_mb:.0f} MB"
            )
        else:
            print(
                f"  Window {i+1}/{len(windows)}: {elapsed:.1f}s, "
                f"WARNING: {expected_file} not found!"
            )

    total_time = time.time() - t0
    print(f"\nDone. {len(windows)} windows in {total_time:.1f}s")

    # Verify
    saved = list(logits_dir.glob("*.safetensors"))
    print(f"Files saved: {len(saved)}")
    if saved:
        # Check first file shape
        data = load_file(str(saved[0]))
        if "log_probs" in data:
            shape = data["log_probs"].shape
            print(f"First file shape: {shape}")
        else:
            print(f"Warning: first file keys: {list(data.keys())}")


def compute_kld(args):
    """Phase compute: load ref + test logits, compute KLD."""
    print("=== Phase: compute ===")

    ref_dir = Path(args.ref_dir)
    test_dirs = [Path(d) for d in args.test_dirs]
    test_names = args.test_names or [str(d) for d in test_dirs]

    # Count ref files
    ref_files = sorted(ref_dir.glob("*.safetensors"), key=lambda p: int(p.stem))
    num_windows = len(ref_files)
    print(f"Reference: {ref_dir} ({num_windows} windows)")

    if num_windows == 0:
        print("Error: no reference logits found")
        sys.exit(1)

    # Verify shape of first ref file
    ref_sample = load_file(str(ref_files[0]))
    ref_shape = ref_sample["log_probs"].shape
    print(f"Reference logit shape: {ref_shape}")
    num_positions_per_window = ref_shape[0]
    vocab_size = ref_shape[1]
    total_positions = num_windows * num_positions_per_window

    results = []

    for test_dir, test_name in zip(test_dirs, test_names):
        print(f"\nComputing KLD: {test_name} vs reference")
        test_files = sorted(test_dir.glob("*.safetensors"), key=lambda p: int(p.stem))

        if len(test_files) != num_windows:
            print(
                f"  Warning: test has {len(test_files)} windows, "
                f"ref has {num_windows}. Using min."
            )
            n = min(len(test_files), num_windows)
        else:
            n = num_windows

        total_kld = 0.0
        all_kld_per_position = []
        total_ref_nll = 0.0  # for perplexity
        count = 0

        nan_positions_total = 0

        for i in range(n):
            ref_data = load_file(str(ref_files[i]))
            test_data = load_file(str(test_files[i]))

            ref_log_probs = ref_data["log_probs"].cuda().float()
            test_log_probs = test_data["log_probs"].cuda().float()

            # Truncate vocab to common size (e.g. VLM 248320 vs text 152064)
            if ref_log_probs.shape[-1] != test_log_probs.shape[-1]:
                min_vocab = min(ref_log_probs.shape[-1], test_log_probs.shape[-1])
                if i == 0:
                    print(
                        f"  Vocab mismatch: ref={ref_log_probs.shape[-1]} "
                        f"test={test_log_probs.shape[-1]} — truncating to {min_vocab}"
                    )
                ref_log_probs = ref_log_probs[:, :min_vocab]
                test_log_probs = test_log_probs[:, :min_vocab]
                # Re-normalize log probs after truncation
                ref_log_probs = ref_log_probs - ref_log_probs.logsumexp(dim=-1, keepdim=True)
                test_log_probs = test_log_probs - test_log_probs.logsumexp(dim=-1, keepdim=True)

            if ref_log_probs.shape != test_log_probs.shape:
                print(
                    f"  Window {i}: shape mismatch "
                    f"ref={ref_log_probs.shape} test={test_log_probs.shape}"
                )
                continue

            # Check for NaN rows (model produced garbage logits)
            ref_has_nan = torch.isnan(ref_log_probs).any(dim=-1)
            test_has_nan = torch.isnan(test_log_probs).any(dim=-1)
            bad_rows = ref_has_nan | test_has_nan
            nan_in_window = bad_rows.sum().item()
            if nan_in_window > 0:
                nan_positions_total += nan_in_window
                if nan_in_window == ref_log_probs.shape[0]:
                    print(f"  Window {i}: ALL {nan_in_window} positions have NaN — skipping")
                    continue
                else:
                    print(f"  Window {i}: {nan_in_window} positions have NaN — excluding them")
                good = ~bad_rows
                ref_log_probs = ref_log_probs[good]
                test_log_probs = test_log_probs[good]

            # KL(ref || test) = sum_x ref(x) * (log ref(x) - log test(x))
            # Using log-space inputs: F.kl_div expects input=log Q, target=log P
            # with log_target=True, computes sum_x exp(log P) * (log P - log Q)
            kld = F.kl_div(
                test_log_probs,  # log Q (test)
                ref_log_probs,   # log P (reference)
                log_target=True,
                reduction="none",
            ).sum(dim=-1)  # [num_positions]

            total_kld += kld.sum().item()
            all_kld_per_position.append(kld.cpu())

            count += kld.numel()

            if (i + 1) % 10 == 0 or i == n - 1:
                running_mean = total_kld / count
                print(f"  Window {i+1}/{n}: running mean KLD = {running_mean:.6f}")

        mean_kld = total_kld / count if count > 0 else float("nan")
        all_kld = torch.cat(all_kld_per_position)
        median_kld = all_kld.median().item()
        p95_kld = all_kld.quantile(0.95).item()
        p99_kld = all_kld.quantile(0.99).item()
        max_kld = all_kld.max().item()

        results.append({
            "name": test_name,
            "mean_kld": mean_kld,
            "median_kld": median_kld,
            "p95_kld": p95_kld,
            "p99_kld": p99_kld,
            "max_kld": max_kld,
            "num_positions": count,
            "nan_positions": nan_positions_total,
            "num_windows": n,
        })

    # Print results table
    print("\n")
    header = (
        f"KLD Evaluation Results "
        f"(ref: {ref_dir.name}, {total_positions:,} target positions)"
    )
    print(header)
    print("=" * len(header))
    print()

    # Table header
    fmt = "{:<40s} {:>10s} {:>12s} {:>10s} {:>10s} {:>10s}"
    print(fmt.format("Model", "Mean KLD", "Median KLD", "P95 KLD", "P99 KLD", "Max KLD"))
    print("-" * 96)

    for r in results:
        print(
            fmt.format(
                r["name"],
                f"{r['mean_kld']:.6f}",
                f"{r['median_kld']:.6f}",
                f"{r['p95_kld']:.6f}",
                f"{r['p99_kld']:.6f}",
                f"{r['max_kld']:.4f}",
            )
        )

    print()
    for r in results:
        nan_note = ""
        if r.get("nan_positions", 0) > 0:
            nan_note = f" ({r['nan_positions']:,} NaN positions excluded)"
        print(f"  {r['name']}: {r['num_positions']:,} positions across {r['num_windows']} windows{nan_note}")

    # Also dump raw results as JSON
    print("\n--- Raw results (JSON) ---")
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="KLD evaluation for quantized models via SGLang",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=["ref", "test", "compute"],
        help="Evaluation phase: ref/test (generate logits) or compute (KLD)",
    )

    # Server connection (for ref/test phases)
    parser.add_argument("--server-url", default="http://localhost:5000")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout (s)")

    # Tokenizer / dataset (for ref/test phases)
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3.5-397B-A17B-FP8",
        help="Tokenizer to use (must be same for ref and test)",
    )
    parser.add_argument(
        "--dataset-config",
        default="wikitext-2-raw-v1",
        help="Wikitext config name",
    )
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--num-windows", type=int, default=100)
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=152064,
        help="Vocabulary size (for storage estimate)",
    )

    # Logits directory (for ref/test phases)
    parser.add_argument(
        "--logits-dir",
        help="Directory where logits are saved by the server patch",
    )

    # Compute phase args
    parser.add_argument("--ref-dir", help="Reference logits directory")
    parser.add_argument(
        "--test-dirs",
        nargs="+",
        help="Test logits directories",
    )
    parser.add_argument(
        "--test-names",
        nargs="+",
        help="Display names for test models",
    )

    args = parser.parse_args()

    if args.phase in ("ref", "test"):
        if not args.logits_dir:
            parser.error("--logits-dir is required for ref/test phases")
        os.makedirs(args.logits_dir, exist_ok=True)
        run_logit_generation(args)
    elif args.phase == "compute":
        if not args.ref_dir:
            parser.error("--ref-dir is required for compute phase")
        if not args.test_dirs:
            parser.error("--test-dirs is required for compute phase")
        compute_kld(args)


if __name__ == "__main__":
    main()
