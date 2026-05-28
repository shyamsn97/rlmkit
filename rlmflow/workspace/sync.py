"""Small workspace sync utilities."""

from __future__ import annotations

import fnmatch
import posixpath
import threading
from pathlib import Path

DEFAULT_EXCLUDES: tuple[str, ...] = (
    ".git/**",
    "__pycache__/**",
    ".pytest_cache/**",
    ".ruff_cache/**",
)

ENGINE_STATE_PATHS: tuple[str, ...] = ("graph.json", "session", "context")

_LOCKS: dict[Path, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def sync_lock_for(path: str | Path) -> threading.RLock:
    """Return a process-local sync lock for a workspace root."""

    root = Path(path).resolve()
    with _LOCKS_GUARD:
        lock = _LOCKS.get(root)
        if lock is None:
            lock = threading.RLock()
            _LOCKS[root] = lock
        return lock


def excluded(rel: str, excludes: tuple[str, ...] = DEFAULT_EXCLUDES) -> bool:
    """Return whether a workspace-relative path should be skipped."""

    rel = rel.strip("/")
    for pattern in excludes:
        pattern = pattern.strip("/")
        if not pattern:
            continue
        if pattern.endswith("/**"):
            base = pattern[:-3].strip("/")
            if rel == base or rel.startswith(base + "/"):
                return True
        if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(
            posixpath.basename(rel), pattern
        ):
            return True
    return False


def engine_state_path(rel: str) -> bool:
    """Return whether ``rel`` is coordinator-owned workspace state."""

    rel = rel.strip("/")
    return any(rel == path or rel.startswith(path + "/") for path in ENGINE_STATE_PATHS)


__all__ = [
    "DEFAULT_EXCLUDES",
    "ENGINE_STATE_PATHS",
    "engine_state_path",
    "excluded",
    "sync_lock_for",
]
