"""Fix Qwen3 reasoning parser: treat <tool_call> as implicit </think> end.

Qwen3.5 sometimes emits <tool_call> inside <think> without closing </think>.
This causes the tool call to be silently dropped. Fix: treat <tool_call> as
implicit end-of-reasoning marker.

Based on PR #35687. Applied as patch because cherry-pick has merge conflicts.
Remove once PR #35687 or equivalent is merged upstream.
"""
import vllm.reasoning.qwen3_reasoning_parser as mod

f = mod.__file__
c = open(f).read()

# Add tool_call handling in the "no </think> found" branch
old = '''\
            if not self.thinking_enabled:
                # Thinking explicitly disabled — treat everything as content.
                return None, model_output
            # Thinking enabled but no </think>: output was truncated.
            # Everything generated so far is reasoning.
            return model_output, None'''

new = '''\
            if not self.thinking_enabled:
                # Thinking explicitly disabled — treat everything as content.
                return None, model_output
            # No </think> — check for implicit reasoning end via <tool_call>.
            # Qwen3.5 sometimes emits <tool_call> inside <think> without
            # closing </think> first. Treat it as implicit end-of-reasoning.
            tool_call_index = model_output.find("<tool_call>")
            if tool_call_index != -1:
                reasoning = model_output[:tool_call_index]
                content = model_output[tool_call_index:]
                return reasoning or None, content or None
            # Thinking enabled but no </think>: output was truncated.
            # Everything generated so far is reasoning.
            return model_output, None'''

assert old in c, 'Pattern not found — maybe already fixed upstream. Remove this patch.'
c = c.replace(old, new)
open(f, 'w').write(c)
print('OK: Qwen3 tool_call implicit reasoning end fix applied to', f)
