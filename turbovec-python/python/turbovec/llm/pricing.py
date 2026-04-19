"""Approximate USD per 1K tokens for common models.

Used to translate cache hits into ``dollars_saved`` in ``SemanticCache.stats()``.
Prices are list-price as of writing and WILL drift — users should override via
``SemanticCache(pricing=...)``.
"""
from __future__ import annotations

from typing import Optional


DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    # (input_per_1k_usd, output_per_1k_usd)
    "gpt-4o":             (0.0025, 0.01),
    "gpt-4o-mini":        (0.00015, 0.0006),
    "gpt-4.1":            (0.002,  0.008),
    "gpt-4.1-mini":       (0.0004, 0.0016),
    "gpt-4.1-nano":       (0.0001, 0.0004),
    "o4-mini":            (0.0011, 0.0044),
    "claude-3-5-sonnet":  (0.003,  0.015),
    "claude-3-5-haiku":   (0.0008, 0.004),
    "claude-sonnet-4":    (0.003,  0.015),
    "claude-sonnet-4-6":  (0.003,  0.015),
    "claude-opus-4":      (0.015,  0.075),
    "claude-opus-4-7":    (0.015,  0.075),
    "claude-haiku-4-5":   (0.001,  0.005),
    "gemini-2.0-flash":   (0.0001, 0.0004),
    "gemini-2.5-pro":     (0.00125, 0.005),
    "gemini-2.5-flash":   (0.0003, 0.0025),
}


def price_for_model(model: str, pricing: Optional[dict[str, tuple[float, float]]] = None) -> tuple[float, float]:
    """Return (input_per_1k, output_per_1k) USD. Matches on longest prefix."""
    table = pricing if pricing is not None else DEFAULT_PRICING
    if model in table:
        return table[model]
    best_key = ""
    for k in table:
        if model.startswith(k) and len(k) > len(best_key):
            best_key = k
    if best_key:
        return table[best_key]
    return (0.0, 0.0)


def estimate_cost_usd(
    model: str,
    tokens_in: int,
    tokens_out: int,
    pricing: Optional[dict[str, tuple[float, float]]] = None,
) -> float:
    pin, pout = price_for_model(model, pricing)
    return (tokens_in / 1000.0) * pin + (tokens_out / 1000.0) * pout


__all__ = ["DEFAULT_PRICING", "price_for_model", "estimate_cost_usd"]
