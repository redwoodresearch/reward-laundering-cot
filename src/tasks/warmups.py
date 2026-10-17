"""Warm-up exchanges for the CoT-control gates. A warm-up pool (data/warmups/<rule>.<version>.jsonl, built by
scripts/data/build_warmup_pool.py) holds the model's own traces, sampled under `warmup_message` (the problem, the answer
format and the rule; no gate wording) and made to follow the rule. Each is shown as one self-contained exchange before
the untouched gate; a question gets the first k of a fixed per-question permutation (`warmup_order`), so k=1 is part of
k=2. The permutation is keyed by the sha256 of the pool file's bytes: never edit a published pool, not even whitespace,
or every question's warm-ups change."""

from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path

from src.tasks.maths import INTEGER_ANSWER_FORMAT

WARMUPS_DIR = Path(__file__).resolve().parents[2] / "data" / "warmups"


@cache
def warmup_pool(rule: str, version: str) -> tuple[list[dict], str]:
    """The pool's rows (file order) and the sha256 of its bytes."""
    data = (WARMUPS_DIR / f"{rule}.{version}.jsonl").read_bytes()
    return [json.loads(line) for line in data.decode().splitlines()], hashlib.sha256(data).hexdigest()


def warmup_message(problem: str, rule: str) -> str:
    """A self-contained solve request with the rule scoped to its own thinking: no turns, no gate."""
    return f"{problem}\n\n{INTEGER_ANSWER_FORMAT}\n\nRequirement: your thinking {rule}."


def warmup_order(rows: list[dict], digest: str, key: str, problem_id: str, problem: str) -> list[dict]:
    """A fixed permutation of the pool for one question (sha256 of pool digest, key and question id), minus any
    exemplar made from that question."""
    usable = [r for r in rows if r["source"]["problem_id"] != problem_id and r["problem"] != problem]
    return sorted(usable, key=lambda r: hashlib.sha256(f"{digest}|{key}|{problem_id}|{r['id']}".encode()).hexdigest())


def assistant(thinking: str, response: str) -> dict:
    return {"role": "assistant", "content": [{"type": "thinking", "thinking": thinking}, {"type": "text", "text": response}]}
