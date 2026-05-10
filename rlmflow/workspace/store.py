"""Shared storage backends for workspace session/context data."""

from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Store(ABC):
    """Backend-neutral object store rooted at one workspace."""

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Return whether ``path`` exists in the store."""

    @abstractmethod
    def read_text(self, path: str) -> str:
        """Read UTF-8 text from ``path``."""

    @abstractmethod
    def write_text(self, path: str, value: str) -> None:
        """Write UTF-8 text to ``path``."""

    @abstractmethod
    def append_text(self, path: str, value: str) -> None:
        """Append UTF-8 text to ``path``."""

    @abstractmethod
    def list(self, prefix: str = "") -> list[str]:
        """List stored paths under ``prefix``."""

    @abstractmethod
    def fork(self, new_location: object) -> Store:
        """Return a deep copy of this store."""

    def read_json(self, path: str) -> Any:
        return json.loads(self.read_text(path))

    def write_json(self, path: str, value: Any) -> None:
        self.write_text(path, json.dumps(value, indent=2))

    def append_jsonl(self, path: str, value: Any) -> None:
        if hasattr(value, "model_dump_json"):
            line = value.model_dump_json()
        else:
            line = json.dumps(value, default=str)
        self.append_text(path, line + "\n")

    def read_jsonl(self, path: str) -> list[Any]:
        if not self.exists(path):
            return []
        return [
            json.loads(line)
            for line in self.read_text(path).splitlines()
            if line.strip()
        ]


class FileStore(Store):
    """Local filesystem implementation of :class:`Store`."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, path: str) -> Path:
        return self.root / path

    def exists(self, path: str) -> bool:
        return self.path(path).exists()

    def read_text(self, path: str) -> str:
        return self.path(path).read_text(encoding="utf-8")

    def write_text(self, path: str, value: str) -> None:
        out = self.path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(value, encoding="utf-8")

    def append_text(self, path: str, value: str) -> None:
        out = self.path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(value)

    def list(self, prefix: str = "") -> list[str]:
        base = self.path(prefix)
        if not base.exists():
            return []
        if base.is_file():
            return [prefix]
        return sorted(
            str(path.relative_to(self.root))
            for path in base.rglob("*")
            if path.is_file()
        )

    def fork(self, new_location: object) -> Store:
        dst = Path(new_location).resolve()
        if dst.exists():
            shutil.rmtree(dst)
        if self.root.exists():
            shutil.copytree(self.root, dst)
        else:
            dst.mkdir(parents=True)
        return FileStore(dst)


class MemoryStore(Store):
    """In-memory :class:`Store` useful for tests and future non-file backends."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def exists(self, path: str) -> bool:
        return path in self.values

    def read_text(self, path: str) -> str:
        return self.values[path]

    def write_text(self, path: str, value: str) -> None:
        self.values[path] = value

    def append_text(self, path: str, value: str) -> None:
        self.values[path] = self.values.get(path, "") + value

    def list(self, prefix: str = "") -> list[str]:
        return sorted(path for path in self.values if path.startswith(prefix))

    def fork(self, new_location: object) -> Store:
        del new_location
        out = MemoryStore()
        out.values = dict(self.values)
        return out


def resolve_backend(
    root: Store | str | Path,
    *,
    legacy_dirs: tuple[str, ...] = (),
) -> tuple[Store, Path | None, bool]:
    """Resolve a ``Store`` or path into ``(store, root, legacy)``.

    Passing a ``Store`` opts in to the new flat workspace layout; passing a
    path keeps the legacy standalone layout rooted at that path. ``legacy_dirs``
    are subdirectories the legacy mode wants pre-created.
    """
    if isinstance(root, Store):
        return root, getattr(root, "root", None), False

    path = Path(root).resolve()
    path.mkdir(parents=True, exist_ok=True)
    for rel in legacy_dirs:
        (path / rel).mkdir(parents=True, exist_ok=True)
    return FileStore(path), path, True


def copy_workspace_root(src_root: Path | None, new_location: object) -> Path:
    """Copy the legacy standalone root into ``new_location`` (replacing it)."""
    dst = Path(new_location).resolve()
    if dst.exists():
        shutil.rmtree(dst)
    if src_root is not None and src_root.exists():
        shutil.copytree(src_root, dst)
    else:
        dst.mkdir(parents=True)
    return dst


def copy_workspace_paths(
    store: Store,
    new_location: object,
    paths: tuple[str, ...],
) -> Store:
    """Copy a subset of files/directories from a workspace ``Store``.

    For ``FileStore`` backends, copies only the named relative paths into the
    new location and returns a ``FileStore`` rooted there. For other backends,
    falls back to the store's own ``fork`` (a deep copy).
    """
    if not isinstance(store, FileStore):
        return store.fork(new_location)

    dst = Path(new_location).resolve()
    dst.mkdir(parents=True, exist_ok=True)
    for rel in paths:
        src = store.path(rel)
        if not src.exists():
            continue
        out = dst / rel
        if src.is_dir():
            if out.exists():
                shutil.rmtree(out)
            shutil.copytree(src, out)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
    return FileStore(dst)


__all__ = [
    "FileStore",
    "MemoryStore",
    "Store",
    "copy_workspace_paths",
    "copy_workspace_root",
    "resolve_backend",
]
