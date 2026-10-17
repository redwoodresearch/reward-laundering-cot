"""Proportions with Wilson intervals; an empty denominator gives None, never a misleading 0."""

from __future__ import annotations

import math


def rate(k: int, n: int, ci: bool = True, z: float = 1.96) -> dict:
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, **({"ci95": None} if ci else {})}
    out = {"k": k, "n": n, "rate": k / n}
    if ci:
        p, d = k / n, 1 + z * z / n
        center, half = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        out["ci95"] = [max(0.0, center - half), min(1.0, center + half)]
    return out


def samples_for_margin(n_items: int, margin: float, z: float = 1.96) -> int:
    """Samples per item so a rate over all of them has a 95% half-width <= `margin` even at 50% (worst case)."""
    return math.ceil(z * z * 0.25 / (margin * margin * n_items))


def eval_precision(per_item: list[list[float]], planned_per_item: int, margin: float, z: float = 1.96) -> dict:
    """Precision of a 0/1 eval on a fixed panel. `half_width`: approximate worst-case 95% half-width for *these* items,
    from the design (None, and `target_met` False, if any item's count differs from planned). `question_spread`:
    z * sd(item means) / sqrt(#items), an approximate diagnostic of how much the rate depends on which items were asked."""
    n_items = len(per_item)
    complete = all(len(xs) == planned_per_item for xs in per_item)
    half_width = z / (2 * math.sqrt(n_items * planned_per_item)) if complete and n_items else None
    means = [sum(xs) / len(xs) for xs in per_item if xs]
    spread = None
    if len(means) > 1:
        mean = sum(means) / len(means)
        spread = z * math.sqrt(sum((m - mean) ** 2 for m in means) / (len(means) - 1) / len(means))
    return {"target_margin": margin, "items": n_items, "planned_per_item": planned_per_item,
            "received": sum(len(xs) for xs in per_item), "items_short": sum(len(xs) < planned_per_item for xs in per_item),
            "half_width": half_width, "target_met": half_width is not None and half_width <= margin + 1e-9,
            "question_spread": spread}
