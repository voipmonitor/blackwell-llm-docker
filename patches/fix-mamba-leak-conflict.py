#!/usr/bin/env python3
"""Resolve cherry-pick conflict for PR #20182 (fix mamba memory leak).

The conflict is in scheduler.py: main added hisparse_coordinator logic
inside the abort handler block, but PR #20182 refactored that block into
_cleanup_waiting_request_resources(). Resolution: keep the PR's refactored
call, and update the helper to include hisparse logic from main.
"""
import sys

path = '/opt/sglang/python/sglang/srt/managers/scheduler.py'
with open(path) as f:
    src = f.read()

# Step 1: Resolve the conflict markers at the abort handler
# Replace the conflict block with the PR's clean call
CONFLICT = '''<<<<<<< HEAD
            # For disaggregation decode mode, the request in the waiting queue has KV cache allocated.
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                if self.enable_hisparse:
                    self.hisparse_coordinator.request_finished(req)
                release_kv_cache(req, self.tree_cache)
            # For disaggregation prefill mode, free the metadata buffer index
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                release_req_to_metadata_buffer(
                    req, self.req_to_metadata_buffer_idx_allocator
                )

            # For mamba radix cache
            if (
                req.mamba_pool_idx is not None
                and self.disaggregation_mode != DisaggregationMode.DECODE
            ):
                release_kv_cache(req, self.tree_cache, is_insert=False)
=======
            self._cleanup_waiting_request_resources(req)
>>>>>>> '''

# Find and remove conflict (the cherry-pick SHA suffix varies)
import re
pattern = re.escape(CONFLICT.split('>>>>>>>')[0]) + r'>>>>>>>[^\n]*\n'
m = re.search(pattern, src)
if m:
    src = src[:m.start()] + '            self._cleanup_waiting_request_resources(req)\n' + src[m.end():]
    print('OK: resolved conflict markers in abort handler')
elif 'self._cleanup_waiting_request_resources(req)' in src and '<<<<<<<' not in src:
    print('OK: conflict already resolved')
else:
    print(f'ERROR: conflict pattern not found in {path}', file=sys.stderr)
    sys.exit(1)

# Step 2: Update _cleanup_waiting_request_resources to include hisparse logic
OLD_CLEANUP = '''    def _cleanup_waiting_request_resources(self, req: Req) -> None:
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            release_kv_cache(req, self.tree_cache)'''

NEW_CLEANUP = '''    def _cleanup_waiting_request_resources(self, req: Req) -> None:
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            if self.enable_hisparse:
                self.hisparse_coordinator.request_finished(req)
            release_kv_cache(req, self.tree_cache)'''

if OLD_CLEANUP in src:
    src = src.replace(OLD_CLEANUP, NEW_CLEANUP)
    print('OK: added hisparse_coordinator to _cleanup_waiting_request_resources')
elif NEW_CLEANUP in src:
    print('OK: hisparse logic already present')
else:
    print(f'ERROR: _cleanup_waiting_request_resources pattern not found', file=sys.stderr)
    sys.exit(1)

with open(path, 'w') as f:
    f.write(src)

# Verify no conflict markers remain
if '<<<<<<<' in src or '>>>>>>>' in src:
    print('ERROR: conflict markers still present!', file=sys.stderr)
    sys.exit(1)

print('OK: PR #20182 conflict resolved')
