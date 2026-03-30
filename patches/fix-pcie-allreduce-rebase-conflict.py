"""Resolve merge conflicts from rebased PCIe allreduce (e4b4d46).

Conflicts:
  layernorm.py — 3 blocks:
    1. else:/elif _is_flashinfer_available: → keep incoming (has guard)
    2. forward_hip/forward_with_allreduce_fusion → keep both (different methods)
    3. hip forward body / allreduce fusion body → keep both
  decode.py — 1 block: imports → keep both
"""
from pathlib import Path
import re
import sys

SGLANG_ROOT = Path("/opt/sglang/python/sglang/srt")


def resolve_file(filepath, strategies):
    """Resolve conflicts in a file with per-block strategies.

    strategies: list of "head", "incoming", or "both" — one per conflict block.
    If fewer strategies than blocks, remaining use "both".
    """
    if not filepath.exists():
        return
    content = filepath.read_text()
    if "<<<<<<< HEAD" not in content:
        print(f"  {filepath.name}: no conflicts")
        return

    pattern = r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]+\n'
    conflicts = list(re.finditer(pattern, content, flags=re.DOTALL))
    print(f"  {filepath.name}: {len(conflicts)} conflict(s)")

    # Replace from end to preserve offsets
    for i, m in enumerate(reversed(conflicts)):
        idx = len(conflicts) - 1 - i
        strategy = strategies[idx] if idx < len(strategies) else "both"
        head = m.group(1)
        incoming = m.group(2)
        if callable(strategy):
            replacement = strategy(head, incoming)
        elif strategy == "head":
            replacement = head
        elif strategy == "incoming":
            replacement = incoming
        else:
            replacement = head + incoming
        content = content[:m.start()] + replacement + content[m.end():]

    if "<<<<<<< HEAD" in content:
        print(f"ERROR: unresolved in {filepath}")
        sys.exit(1)
    filepath.write_text(content)
    print(f"  {filepath.name}: resolved ({strategies})")


# layernorm.py: 3 conflicts
# 1. else:/elif guard → incoming (adds _is_flashinfer_available check)
# 2+3. Two methods sharing a signature — HEAD=forward_hip, incoming=forward_with_allreduce_fusion
#    Conflict 2 is the method name, conflict 3 is the body.
#    Strategy: keep HEAD for both (preserves forward_hip), then manually
#    inject forward_with_allreduce_fusion from incoming of conflict 3.

def layernorm_strategy_2(head, incoming):
    """Keep forward_hip name from HEAD."""
    return head

def layernorm_strategy_3(head, incoming):
    """Keep forward_hip body (HEAD), then add forward_with_allreduce_fusion (incoming) as new method."""
    return head + "\n    def forward_with_allreduce_fusion(\n        self,\n        x: torch.Tensor,\n        residual: Optional[torch.Tensor] = None,\n        post_residual_addition: Optional[torch.Tensor] = None,\n    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:\n" + incoming

resolve_file(SGLANG_ROOT / "layers/layernorm.py", ["incoming", layernorm_strategy_2, layernorm_strategy_3])

# decode.py: 1 conflict — imports → both
resolve_file(SGLANG_ROOT / "disaggregation/decode.py", ["both"])

# Catch-all
for f in SGLANG_ROOT.rglob("*.py"):
    try:
        if "<<<<<<< HEAD" in f.read_text():
            print(f"ERROR: unresolved conflict in {f}")
            sys.exit(1)
    except Exception:
        pass
print("OK: PCIe allreduce conflicts resolved")
