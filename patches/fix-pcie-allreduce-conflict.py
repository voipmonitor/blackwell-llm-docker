"""Resolve merge conflicts from lukealonso cherry-picks and patch PCIe allreduce.

Handles conflicts from:
1. PCIe allreduce (5bb89b03): custom_all_reduce.py + utils patch
2. b12x integration (d4917b2): fp4_utils.py, qwen2_moe.py
"""
from pathlib import Path
import re

SGLANG_ROOT = Path("/opt/sglang/python/sglang/srt")


def resolve_conflict(filepath, strategy="head"):
    """Resolve git conflict markers in a file.

    strategy:
      "head"     — keep HEAD side, drop incoming
      "incoming" — keep incoming side, drop HEAD
      "both"     — keep both (HEAD first, then incoming)
      callable   — custom function(head_code, incoming_code) -> replacement
    """
    if not filepath.exists():
        return False

    content = filepath.read_text()
    if "<<<<<<< HEAD" not in content:
        return False

    def replacer(m):
        head = m.group(1)
        incoming = m.group(2)
        if strategy == "head":
            return head
        elif strategy == "incoming":
            return incoming
        elif strategy == "both":
            return head + incoming
        elif callable(strategy):
            return strategy(head, incoming)
        return head

    pattern = r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]+\n'
    content = re.sub(pattern, replacer, content, flags=re.DOTALL)

    if "<<<<<<< HEAD" in content:
        print(f"ERROR: unresolved conflict markers remain in {filepath}")
        exit(1)

    filepath.write_text(content)
    return True


# --- 1. custom_all_reduce.py: keep HEAD (self.group = group) ---

ar_file = SGLANG_ROOT / "distributed/device_communicators/custom_all_reduce.py"
if resolve_conflict(ar_file, strategy="head"):
    print(f"OK: resolved conflict in {ar_file.name} (kept HEAD)")
else:
    print(f"OK: no conflict in {ar_file.name}")


# --- 2. custom_all_reduce_utils.py: allow PCIe topologies ---

utils_file = SGLANG_ROOT / "distributed/device_communicators/custom_all_reduce_utils.py"
if utils_file.exists():
    utils_content = utils_file.read_text()

    OLD_PCIE_BLOCK = """\
    if world_size > 2 and not full_nvlink:
        logger.warning(
            f"{cls_name} is disabled because it's not supported on"
            " more than two PCIe-only GPUs. To silence this warning, "
            "specify disable_custom_all_reduce=True explicitly."
        )
        return"""

    NEW_PCIE_BLOCK = """\
    if world_size > 2 and not full_nvlink:
        logger.info(
            "PCIe topology detected with P2P support. "
            "Enabling custom allreduce for small messages (max 64KB)."
        )"""

    if OLD_PCIE_BLOCK in utils_content:
        utils_content = utils_content.replace(OLD_PCIE_BLOCK, NEW_PCIE_BLOCK)
        utils_file.write_text(utils_content)
        print(f"OK: patched {utils_file.name} — PCIe allreduce enabled")
    elif "PCIe topology detected" in utils_content:
        print(f"OK: {utils_file.name} already patched")
    else:
        print(f"WARNING: PCIe rejection block not found in {utils_file.name}")


# --- 3. fp4_utils.py: keep incoming (adds is_b12x), drop is_flashinfer_cutedsl ---

fp4_file = SGLANG_ROOT / "layers/quantization/fp4_utils.py"


def fp4_strategy(head, incoming):
    """Keep is_b12x() from incoming, drop is_flashinfer_cutedsl()."""
    # Extract only is_b12x method from incoming
    lines = []
    for line in incoming.splitlines(keepends=True):
        if "cutedsl" in line.lower() or "CUTEDSL" in line:
            continue
        lines.append(line)
    return head + "".join(lines)


if resolve_conflict(fp4_file, strategy=fp4_strategy):
    print(f"OK: resolved conflict in {fp4_file.name} (added is_b12x)")
else:
    print(f"OK: no conflict in {fp4_file.name}")


# --- 4. qwen2_moe.py: keep incoming (expanded imports) ---

qwen2_file = SGLANG_ROOT / "models/qwen2_moe.py"
if resolve_conflict(qwen2_file, strategy="incoming"):
    print(f"OK: resolved conflict in {qwen2_file.name} (expanded imports)")
else:
    print(f"OK: no conflict in {qwen2_file.name}")


# --- 5. attention_registry.py: keep incoming (adds b12x to Blackwell assert) ---

attn_reg_file = SGLANG_ROOT / "layers/attention/attention_registry.py"
if resolve_conflict(attn_reg_file, strategy="incoming"):
    print(f"OK: resolved conflict in {attn_reg_file.name} (b12x in Blackwell assert)")
else:
    print(f"OK: no conflict in {attn_reg_file.name}")


# --- Catch-all: scan for any remaining conflict markers ---

for py_file in SGLANG_ROOT.rglob("*.py"):
    try:
        if "<<<<<<< HEAD" in py_file.read_text():
            print(f"WARNING: unresolved conflict in {py_file}")
    except Exception:
        pass
