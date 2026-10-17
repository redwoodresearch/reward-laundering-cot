"""Run-artifact file helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> tuple[list[dict], str]:
    """Rows and the sha256 of exactly the bytes they were parsed from (one read, one snapshot)."""
    data = path.read_bytes()
    return [json.loads(line) for line in data.decode().splitlines()], hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Atomic: readers see the old file or the new one, never a partial write."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Atomic too, written row by row (eval files can be large)."""
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)
