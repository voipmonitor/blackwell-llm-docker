#!/usr/bin/env python3
"""Fix merge conflicts and empty if blocks in modelopt_quant.py.

PR #21314 (cutlass backend) + b12x cherry-pick can leave:
1. Git merge conflict markers (<<<<<<< HEAD ... >>>>>>> ...)
2. Empty if blocks where one side of the conflict was removed

This script resolves both issues.
"""
import re

path = 'python/sglang/srt/layers/quantization/modelopt_quant.py'
with open(path) as f:
    src = f.read()

# 1. Resolve git merge conflict markers: keep BOTH sides
src = re.sub(
    r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]+\n',
    lambda m: m.group(1) + m.group(2),
    src,
    flags=re.DOTALL,
)

# 2. Fix empty if blocks by adding 'pass'
# Pattern: an if/elif with ): followed immediately by another if/elif/else
# without any indented body in between
lines = src.split('\n')
fixed = []
i = 0
while i < len(lines):
    fixed.append(lines[i])
    # Check if current line ends an if condition with ):
    stripped = lines[i].rstrip()
    if stripped.endswith('):') and i + 1 < len(lines):
        # Get indentation of current line
        indent = len(lines[i]) - len(lines[i].lstrip())
        next_line = lines[i + 1] if i + 1 < len(lines) else ''
        next_stripped = next_line.lstrip()
        next_indent = len(next_line) - len(next_line.lstrip())
        # If next line is at same or lower indent and starts with if/elif/else/for/return/def
        # then the if block is empty — add pass
        if next_stripped and next_indent <= indent and \
           (next_stripped.startswith(('if ', 'elif ', 'else:', 'for ', 'return ', 'def ', 'class '))):
            fixed.append(' ' * (indent + 4) + 'pass')
    i += 1

src = '\n'.join(fixed)

with open(path, 'w') as f:
    f.write(src)

# Verify it parses
try:
    compile(src, path, 'exec')
    print(f'OK: {path} is syntactically valid')
except SyntaxError as e:
    print(f'WARNING: {path} still has syntax error: {e}')
    # Try to show context
    if e.lineno:
        context = src.split('\n')[max(0,e.lineno-3):e.lineno+2]
        for j, line in enumerate(context):
            print(f'  {e.lineno-2+j}: {line}')
