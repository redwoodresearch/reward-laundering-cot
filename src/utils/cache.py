"""File-per-entry async cache keyed by a hash of the JSON key. Publication is atomic (temp file + hardlink; a racing
loser reads the winner's value) but not single-flight: two concurrent misses both compute. Never edit entries; to
invalidate them after a code change, bump the uuid inside the key."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import aiofiles


class FileCache:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: Any) -> Path:
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
        return self.cache_dir / f"{digest}.cache"

    async def _publish(self, path: Path, data: str) -> bool:
        """Atomically create `path` with `data`; False if it already existed."""
        fd, tmp = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        try:
            os.close(fd)
            async with aiofiles.open(tmp, "w", encoding="utf-8") as f:
                await f.write(data)
            os.link(tmp, path)
            return True
        except FileExistsError:
            return False
        finally:
            os.unlink(tmp)

    async def aget(self, key: Any) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        async with aiofiles.open(path, encoding="utf-8") as f:
            return json.loads(await f.read())

    async def aget_or_compute_set(self, key: Any, compute_fn: Callable[[], Awaitable[Any]]) -> Any:
        """The cached value, else compute and publish it; if another process published first, use theirs."""
        cached = await self.aget(key)
        if cached is not None:
            return cached
        value = await compute_fn()
        if await self._publish(self._path(key), json.dumps(value)):
            return value
        return await self.aget(key)
