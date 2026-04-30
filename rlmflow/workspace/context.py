"""Context payloads — task/data variables injected into the REPL."""

from __future__ import annotations

import json
import re
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

CONTEXT_VARIABLE_PROMPT = """
**Context variable:**
- `CONTEXT.info()` — metadata about the current task/data payload.
- `CONTEXT.read(start=0, end=None)` — read a character slice.
- `CONTEXT.lines(start=0, end=None)` — read line range `start:end`.
- `CONTEXT.line_count()` — count lines.
- `CONTEXT.grep(pattern, max_results=50)` — search by regex.

Use `CONTEXT` for long task input. Do not print the whole context; inspect
samples, split it into chunks, delegate chunk work, and aggregate structured
results.
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
    def fork(self, new_location: object) -> "Context":
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

    def context_prompt_hint(self, *, agent_id: str | None = None) -> str:
        contexts = self.list_contexts(agent_id=agent_id)
        if not contexts:
            return ""
        context_list = "\n".join(f"- `{key}`" for key in contexts)
        return (
            CONTEXT_VARIABLE_PROMPT.strip()
            + "\n\nAvailable context payload keys:\n"
            + context_list
        )


class FileContext(Context):
    """Filesystem payload store under ``workspace/context``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _context_paths(self, key: str, *, agent_id: str) -> tuple[Path, Path]:
        base = self.root / _agent_path(agent_id)
        safe = _safe_name(key)
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
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        meta = {
            "key": key,
            "agent_id": agent_id,
            "chars": len(value),
            "metadata": metadata or {},
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def read(self, key: str = "context", *, agent_id: str = "root") -> str:
        for aid in (agent_id, "root"):
            path, _ = self._context_paths(key, agent_id=aid)
            if path.exists():
                return path.read_text(encoding="utf-8")
        raise KeyError(f"context {key!r} not found for {agent_id!r}")

    def list_contexts(self, *, agent_id: str | None = None) -> list[str]:
        agent_ids = [agent_id] if agent_id else ["root"]
        if agent_id and agent_id != "root":
            agent_ids.append("root")
        keys: set[str] = set()
        for aid in agent_ids:
            if aid is None:
                continue
            base = self.root / _agent_path(aid)
            if base.exists():
                keys.update(p.stem for p in base.glob("*.txt"))
        return sorted(keys)

    def fork(self, new_location: object) -> Context:
        dst = Path(new_location).resolve()
        if dst.exists():
            shutil.rmtree(dst)
        if self.root.exists():
            shutil.copytree(self.root, dst)
        else:
            dst.mkdir(parents=True)
        return FileContext(dst)


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


__all__ = ["Context", "ContextVariable", "FileContext", "InMemoryContext"]
