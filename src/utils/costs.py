"""Tinker cost tracking from the dated snapshot of the published model list (pricing/tinker_models_<date>.json, which
also gives each model's context window), else pricing/tinker.json for models it lacks. Runs keep the costs they
recorded. Unknown models raise; provisional rates are labelled."""

from __future__ import annotations

import atexit
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from src.utils.provenance import git_commit

PRICING_FILE = Path(__file__).parent / "pricing" / "tinker.json"
TINKER_PRICING_PER_MILLION: dict[str, dict] = json.loads(PRICING_FILE.read_text())["models"]
PUBLISHED_FILE = Path(__file__).parent / "pricing" / "tinker_models_20261008.json"  # tinker-docs models.json
PUBLISHED: dict[str, dict] = {m["tinker_id"]: m for m in json.loads(PUBLISHED_FILE.read_text())}
def require_price(model: str) -> dict:
    """$ per million tokens {prefill, sample, train}: the published snapshot's list prices (any limited-time discount
    included, cached-prefill discounts excluded), else pricing/tinker.json."""
    if model in PUBLISHED:
        return {k: float(PUBLISHED[model][k].lstrip("$")) for k in ("prefill", "sample", "train")}
    if model in TINKER_PRICING_PER_MILLION:
        return TINKER_PRICING_PER_MILLION[model]
    raise KeyError(f"Tinker model {model!r} is in neither {PRICING_FILE} nor {PUBLISHED_FILE}")


def context_window(model: str) -> int | None:
    """The model's published context window in tokens (e.g. "64K" -> 65536), or None if not published."""
    context = PUBLISHED.get(model, {}).get("context")
    return int(context.removesuffix("K")) * 1024 if context else None


class CostTracker:
    """Accumulates this run's Tinker cost and appends one summary row to `cost_file` at exit."""

    def __init__(self, cost_file: Path, run_description: str | None = None):
        self.cost_file = Path(cost_file)
        self.run_description = run_description
        self.start_time = datetime.now(UTC)
        self.git_commit = git_commit()  # at launch: HEAD may move before the row is written at exit
        self.run_cost = 0.0
        self.cost_by_model: dict[str, float] = defaultdict(float)
        self.provisional_models: set[str] = set()
        atexit.register(self._save_on_exit)

    def add_tinker_cost(self, model: str, prefill_tokens: int = 0, sample_tokens: int = 0, train_tokens: int = 0) -> None:
        pricing = require_price(model)
        cost = (prefill_tokens * pricing["prefill"] + sample_tokens * pricing["sample"]
                + train_tokens * pricing["train"]) / 1_000_000
        if pricing.get("provisional"):
            self.provisional_models.add(model)
        self.run_cost += cost
        self.cost_by_model[model] += cost

    def summary(self) -> str:
        text = f"${self.run_cost:.4f} estimate"  # list prices, no cached-prefill discount: not an invoice
        if self.provisional_models:
            text += f" (provisional rates for {', '.join(sorted(self.provisional_models))})"
        return text

    def _save_on_exit(self) -> None:
        if self.run_cost == 0:
            return
        entry = {
            "start_timestamp": self.start_time.isoformat(),
            "end_timestamp": datetime.now(UTC).isoformat(),
            "full_args": sys.argv,
            "git_commit": self.git_commit,
            "description": self.run_description,
            "run_cost": self.run_cost,
            "cost_by_model": dict(self.cost_by_model),
            "provisional_models": sorted(self.provisional_models),
        }
        self.cost_file.parent.mkdir(parents=True, exist_ok=True)
        with self.cost_file.open("a") as f:
            f.write(json.dumps(entry) + "\n")
