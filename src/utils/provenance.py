"""The source commit and working-tree state recorded with every run."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str | None:
    """Output of a git query on this repository (whatever the cwd), or None if it fails: provenance never blocks."""
    try:
        result = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def git_commit() -> str | None:
    out = _git("rev-parse", "HEAD")
    return out.strip() if out is not None else None


def git_dirty() -> bool | None:
    """Whether the working tree had uncommitted changes (incl. untracked files); None if git could not say."""
    out = _git("status", "--porcelain")
    return None if out is None else bool(out.strip())
