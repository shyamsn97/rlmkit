"""Context abstraction — durable scratchpad for agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import PurePosixPath

from .runtime import Runtime


class Context(ABC):
    """Durable scratchpad that persists across REPL turns.

    Agents interact via read_context() and append_context() tools.
    Each child gets an isolated clone.
    """

    @abstractmethod
    def read(self) -> str:
        """Return the full context contents."""

    @abstractmethod
    def append(self, text: str) -> None:
        """Append text to the context."""

    @abstractmethod
    def write(self, text: str) -> None:
        """Overwrite the context with new text."""

    @abstractmethod
    def clone(self, agent_id: str) -> Context:
        """Create an isolated context for a child agent."""


class FileContext(Context):
    """Context backed by a single file in the runtime workspace."""

    def __init__(self, path: str, runtime: Runtime) -> None:
        self.path = path
        self.runtime = runtime

    def read(self) -> str:
        try:
            return self.runtime.read_file(self.path)
        except FileNotFoundError:
            return ""

    def append(self, text: str) -> None:
        self.runtime.append_file(self.path, text)

    def write(self, text: str) -> None:
        self.runtime.write_file(self.path, text)

    def clone(self, agent_id: str) -> FileContext:
        p = PurePosixPath(self.path)
        child_path = str(p.parent / agent_id / p.name)
        return FileContext(child_path, self.runtime)
