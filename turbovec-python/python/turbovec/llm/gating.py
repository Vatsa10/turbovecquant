"""Rules for when NOT to cache: tool calls, non-determinism, response-format objects."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateDecision:
    cache: bool
    reason: str = ""


def should_cache(
    messages: list[dict],
    kwargs: dict,
    *,
    cache_nondeterministic: bool = False,
    allow_tools: bool = False,
) -> GateDecision:
    temp = kwargs.get("temperature")
    if temp is not None and temp > 0 and not cache_nondeterministic:
        return GateDecision(False, f"temperature={temp} (non-deterministic)")

    if not allow_tools:
        if kwargs.get("tools") or kwargs.get("functions") or kwargs.get("tool_choice"):
            return GateDecision(False, "tools/functions requested (side-effects)")
        for m in messages:
            if m.get("tool_calls") or m.get("function_call") or m.get("role") == "tool":
                return GateDecision(False, "messages contain tool_calls/function_call")

    fmt = kwargs.get("response_format")
    if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
        # json_schema is deterministic-enough, allow it. Kept for extension point.
        pass

    return GateDecision(True, "ok")


__all__ = ["GateDecision", "should_cache"]
