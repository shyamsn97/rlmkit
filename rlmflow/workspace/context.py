"""Context payloads — task/data variables injected into the REPL."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from rlmflow.workspace.store import (
    FileStore,
    Store,
    copy_workspace_paths,
    copy_workspace_root,
    resolve_backend,
)

CONTEXT_VARIABLE_PROMPT = """
**Context variable:**

`CONTEXT` is the read-only data slot for *this* agent — always present, possibly empty.
The query lives in your messages, not here. `CONTEXT` is just the data to operate on.
**Size it up first**: `CONTEXT.info()["chars"] == 0` means no payload — work from the query.

API:
- `CONTEXT.info()` / `CONTEXT.line_count()` — measure.
- `CONTEXT.read(start=0, end=None)` — char slice.
- `CONTEXT.lines(start=0, end=None)` — line slice.
- `CONTEXT.grep(pattern, max_results=50)` — regex; returns `lineno:line` rows.
- `CONTEXT.fork()` — snapshot for handing to a child (isolated copy of your view).

For long `CONTEXT`, don't `print` it whole (output is truncated). Sample, chunk,
delegate one sub-agent per chunk with a structured reply, aggregate in the parent.
"""


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "context"


def _agent_path(agent_id: str) -> Path:
    parts = [p for p in agent_id.split(".") if p and p != "root"]
    return Path(*parts) if parts else Path()


class ContextVariable:
    """Lazy handle over the current task/data payload."""

    def __init__(
        self,
        context: Context,
        *,
        agent_id: str = "root",
        key: str = "context",
    ) -> None:
        self.store = context
        self.agent_id = agent_id
        self.key = key

    def info(self) -> dict[str, Any]:
        return self.store.info(self.key, agent_id=self.agent_id)

    def read(self, start: int = 0, end: int | None = None) -> str:
        return self.store.read(self.key, agent_id=self.agent_id)[start:end]

    def lines(self, start: int = 0, end: int | None = None) -> str:
        lines = self.store.read(self.key, agent_id=self.agent_id).splitlines(True)
        return "".join(lines[start:end])

    def line_count(self) -> int:
        return len(self.store.read(self.key, agent_id=self.agent_id).splitlines())

    def grep(self, pattern: str, *, max_results: int = 50) -> str:
        compiled = re.compile(pattern)
        matches: list[str] = []
        text = self.store.read(self.key, agent_id=self.agent_id)
        for idx, line in enumerate(text.splitlines(), start=1):
            if compiled.search(line):
                matches.append(f"{idx}:{line}")
                if len(matches) >= max_results:
                    break
        return "\n".join(matches)

    def fork(self) -> str:
        """Snapshot this ``CONTEXT`` for handoff to a child agent.

        Use as ``delegate(name, query, CONTEXT.fork())`` — the child
        receives an isolated copy of the parent's payload. Equivalent
        to :meth:`read` with no args; the name conveys intent
        ("inherit my view, own your own slot"). Reach for it only
        when the child genuinely needs the same data the parent saw
        (reviewers, auditors, retry-after-failure). Default delegation
        should pass a fresh slice or ``""``.
        """
        return self.store.read(self.key, agent_id=self.agent_id)


class Context(ABC):
    """Store task/data payloads exposed to the REPL as ``context``."""

    @abstractmethod
    def write(
        self,
        key: str,
        value: str,
        *,
        agent_id: str = "root",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError("This context backend does not support blobs.")

    def read(self, key: str = "context", *, agent_id: str = "root") -> str:
        raise KeyError(f"context {key!r} not found for {agent_id!r}")

    def list_contexts(self, *, agent_id: str | None = None) -> list[str]:
        return []

    @abstractmethod
    def fork(self, new_location: object) -> Context:
        """Return a deep copy of this context payload store."""

    def info(
        self,
        key: str = "context",
        *,
        agent_id: str = "root",
    ) -> dict[str, Any]:
        text = self.read(key, agent_id=agent_id)
        return {
            "key": key,
            "agent_id": agent_id,
            "chars": len(text),
            "approx_tokens": len(text) // 4,
            "lines": len(text.splitlines()),
        }


class FileContext(Context):
    """Store-backed context persistence.

    ``FileContext(path)`` preserves the legacy ``<key>.txt`` +
    ``<key>.json`` layout rooted at ``path``. ``FileContext(store)``
    uses the flat workspace layout under ``context/<agent-id>/``.
    """

    def __init__(self, root: Store | str | Path) -> None:
        self.store, self.root, self.legacy = resolve_backend(root)

    def _context_paths(self, key: str, *, agent_id: str) -> tuple[Path, Path]:
        safe = _safe_name(key)
        if self.legacy:
            base = _agent_path(agent_id)
            return base / f"{safe}.txt", base / f"{safe}.json"
        base = Path("context") / _safe_name(agent_id)
        if safe == "context":
            return base / "context.txt", base / "context_metadata.json"
        return base / f"{safe}.txt", base / f"{safe}_metadata.json"

    def _legacy_context_paths(self, key: str, *, agent_id: str) -> tuple[Path, Path]:
        safe = _safe_name(key)
        base = Path("context") / _agent_path(agent_id)
        return base / f"{safe}.txt", base / f"{safe}.json"

    def write(
        self,
        key: str,
        value: str,
        *,
        agent_id: str = "root",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        path, meta_path = self._context_paths(key, agent_id=agent_id)
        self.store.write_text(str(path), value)
        meta = {
            "key": key,
            "agent_id": agent_id,
            "chars": len(value),
            "metadata": metadata or {},
        }
        self.store.write_json(str(meta_path), meta)

    def read(self, key: str = "context", *, agent_id: str = "root") -> str:
        for aid in (agent_id, "root"):
            path, _ = self._context_paths(key, agent_id=aid)
            if self.store.exists(str(path)):
                return self.store.read_text(str(path))
            if not self.legacy:
                legacy_path, _ = self._legacy_context_paths(key, agent_id=aid)
                if self.store.exists(str(legacy_path)):
                    return self.store.read_text(str(legacy_path))
        raise KeyError(f"context {key!r} not found for {agent_id!r}")

    def list_contexts(self, *, agent_id: str | None = None) -> list[str]:
        agent_ids = [agent_id] if agent_id else ["root"]
        if agent_id and agent_id != "root":
            agent_ids.append("root")
        keys: set[str] = set()
        for aid in agent_ids:
            if aid is None:
                continue
            if self.legacy:
                base = _agent_path(aid)
                if isinstance(self.store, FileStore):
                    base_path = self.store.path(str(base) if str(base) != "." else "")
                    paths = (
                        [
                            str(path.relative_to(self.store.root))
                            for path in base_path.glob("*.txt")
                        ]
                        if base_path.exists()
                        else []
                    )
                else:
                    prefix = str(base) if str(base) != "." else ""
                    paths = self.store.list(prefix)
            else:
                base = Path("context") / _safe_name(aid)
                paths = self.store.list(str(base))
                legacy_base = Path("context") / _agent_path(aid)
                paths.extend(self.store.list(str(legacy_base)))
            for path in paths:
                if path.endswith(".txt"):
                    keys.add(Path(path).stem)
        return sorted(keys)

    def fork(self, new_location: object) -> Context:
        if self.legacy:
            return FileContext(copy_workspace_root(self.root, new_location))
        return FileContext(copy_workspace_paths(self.store, new_location, ("context",)))


class InMemoryContext(Context):
    """Process-local payload store for runs without a Workspace."""

    def __init__(self) -> None:
        self.blobs: dict[tuple[str, str], str] = {}

    def write(
        self,
        key: str,
        value: str,
        *,
        agent_id: str = "root",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        del metadata
        self.blobs[(agent_id, key)] = value

    def read(self, key: str = "context", *, agent_id: str = "root") -> str:
        for aid in (agent_id, "root"):
            if (aid, key) in self.blobs:
                return self.blobs[(aid, key)]
        raise KeyError(f"context {key!r} not found for {agent_id!r}")

    def list_contexts(self, *, agent_id: str | None = None) -> list[str]:
        agent_ids = [agent_id] if agent_id else ["root"]
        if agent_id and agent_id != "root":
            agent_ids.append("root")
        return sorted({key for aid, key in self.blobs if aid in agent_ids})

    def fork(self, new_location: object) -> Context:
        del new_location
        out = InMemoryContext()
        out.blobs = dict(self.blobs)
        return out


__all__ = [
    "CONTEXT_VARIABLE_PROMPT",
    "Context",
    "ContextVariable",
    "FileContext",
    "InMemoryContext",
]
