"""Patch SGLang to capture full logits for KLD evaluation.

Inserts a hook in LogitsProcessor.forward that saves full log-softmax
logits to disk when SGLANG_KLD_SAVE_DIR is set. Only rank 0 saves to
avoid duplicate writes from tensor-parallel workers.

Usage:
    python patches/sglang-kld-logit-capture.py   # apply patch
    SGLANG_KLD_SAVE_DIR=/tmp/kld_ref python -m sglang.launch_server ...

Each prefill request saves one safetensors file with key "log_probs"
of shape [N, vocab_size] in float16.
"""
from pathlib import Path

# Find logits_processor.py without importing sglang (avoids transitive import errors)
candidates = [
    Path("/opt/sglang/python/sglang/srt/layers/logits_processor.py"),
    Path("/sgl-workspace/sglang/python/sglang/srt/layers/logits_processor.py"),
]
# Also search venv site-packages
for p in Path("/opt/venv").rglob("sglang/srt/layers/logits_processor.py"):
    candidates.append(p)

f = None
for p in candidates:
    if p.exists():
        f = str(p)
        break

assert f is not None, "Could not find sglang logits_processor.py"
c = open(f).read()

# --- Inject the _kld_maybe_save helper at the top of the file ---

helper = '''
# --- KLD logit capture patch (injected) ---
import os as _kld_os
import threading as _kld_threading

_kld_lock = _kld_threading.Lock()
_kld_counter = 0

def _kld_maybe_save(input_logits):
    """Save full log-softmax logits to disk for KLD evaluation."""
    global _kld_counter
    save_dir = _kld_os.environ.get("SGLANG_KLD_SAVE_DIR")
    if not save_dir:
        return
    # Only save from rank 0 to avoid duplicates across TP workers
    try:
        import torch.distributed
        rank = torch.distributed.get_rank()
    except (RuntimeError, ValueError):
        rank = 0
    if rank != 0:
        return
    # Trim TP padding columns to actual vocab size.
    # With TP, logits are padded to a multiple of tp_size. The padding columns
    # are garbage and must be removed before log_softmax.
    import torch
    import torch.nn.functional as F
    vocab_size = int(_kld_os.environ.get("SGLANG_KLD_VOCAB_SIZE", "152064"))
    logits = input_logits[:, :vocab_size].float()
    log_probs = F.log_softmax(logits, dim=-1).half()
    from safetensors.torch import save_file
    with _kld_lock:
        idx = _kld_counter
        _kld_counter += 1
    path = _kld_os.path.join(save_dir, f"{idx}.safetensors")
    save_file({"log_probs": log_probs.cpu()}, path)
    print(f"[KLD] Saved logits {log_probs.shape} to {path}")
# --- End KLD logit capture patch ---
'''

# Insert helper after the last top-level import, before first class/function
# Try multiple anchors to handle different SGLang versions
import re
m = re.search(r'^from sglang\.srt\.utils[^\n]*is_npu[^\n]*$', c, re.MULTILINE)
assert m, 'Anchor not found: expected "from sglang.srt.utils... import is_npu" — SGLang version may have changed.'
anchor_line_end = c.index('\n', m.end()) + 1
c = c[:anchor_line_end] + helper + c[anchor_line_end:]

# --- Patch the non-chunked path in forward() ---
# Insert _kld_maybe_save(input_logits) between the indexing and del

old_nonchunked = '''\
            input_logits = logits[input_logprob_indices]
            del logits

            logprobs_result = self.process_input_logprobs(input_logits, logits_metadata)'''

new_nonchunked = '''\
            input_logits = logits[input_logprob_indices]
            del logits
            _kld_maybe_save(input_logits)

            logprobs_result = self.process_input_logprobs(input_logits, logits_metadata)'''

assert old_nonchunked in c, (
    'Non-chunked path pattern not found — SGLang version may have changed. '
    'Remove or update this patch.'
)
c = c.replace(old_nonchunked, new_nonchunked)

open(f, 'w').write(c)
print('OK: KLD logit capture patch applied to', f)
