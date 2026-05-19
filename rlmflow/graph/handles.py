"""REPL protocol handles emitted by ``rlm_delegate()`` and ``rlm_wait()``.

These are transient objects the engine inspects to decide whether to
suspend an agent. They are not stored on the graph; they live alongside
the data model because they bridge agent code and the scheduler.
"""

from __future__ import annotations


class ChildHandle:
    """Opaque reference returned by ``rlm_delegate()``, passed to ``rlm_wait()``."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id

    def __repr__(self) -> str:
        return f"ChildHandle({self.agent_id!r})"

    def to_dict(self) -> dict:
        return {"child_handle": self.agent_id}

    @classmethod
    def from_dict(cls, data: dict) -> ChildHandle:
        return cls(data["child_handle"])


class WaitRequest:
    """Yielded by ``rlm_wait()`` to request suspension until children finish."""

    def __init__(self, agent_ids: list[str]) -> None:
        self.agent_ids = agent_ids

    def __repr__(self) -> str:
        return f"WaitRequest({self.agent_ids!r})"

    def to_dict(self) -> dict:
        return {"wait_request": self.agent_ids}

    @classmethod
    def from_dict(cls, data: dict) -> WaitRequest:
        return cls(data["wait_request"])


__all__ = ["ChildHandle", "WaitRequest"]
