"""RLMState, StepEvent hierarchy, Status enum, and ChildHandle."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

# ── Status ────────────────────────────────────────────────────────────


class Status(str, Enum):
    WAITING = "waiting"
    HAS_REPLY = "has_reply"
    SUPERVISING = "supervising"
    FINISHED = "finished"


# ── StepEvent hierarchy ──────────────────────────────────────────────


class StepEvent(BaseModel):
    agent_id: str
    iteration: int


class LLMReply(StepEvent):
    text: str
    code: str | None = None


class CodeExec(StepEvent):
    code: str
    output: str
    suspended: bool = False


class ChildStep(StepEvent):
    child_events: list[StepEvent] = []
    all_done: bool = False
    exec_output: str | None = None
    # True when done() was called in the resumed exec after children returned.
    # Purely for event-stream observers — state.finished already covers this.
    agent_finished: bool = False


class NoCodeBlock(StepEvent):
    text: str


# ── ChildHandle ──────────────────────────────────────────────────────


class ChildHandle:
    """Opaque reference returned by delegate(), passed to wait()."""

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
    """Yielded by wait() to request suspension until children finish."""

    def __init__(self, agent_ids: list[str]) -> None:
        self.agent_ids = agent_ids

    def __repr__(self) -> str:
        return f"WaitRequest({self.agent_ids!r})"

    def to_dict(self) -> dict:
        return {"wait_request": self.agent_ids}

    @classmethod
    def from_dict(cls, data: dict) -> WaitRequest:
        return cls(data["wait_request"])


# ── RLMState ─────────────────────────────────────────────────────────


class RLMState(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent_id: str = ""
    task: str = ""
    status: Status = Status.WAITING
    iteration: int = 0
    config: dict = {}

    event: StepEvent | None = None
    messages: list[dict] = []
    last_reply: str | None = None
    result: str | None = None

    children: list[RLMState] = []
    waiting_on: list[str] = []

    @property
    def finished(self) -> bool:
        return self.status == Status.FINISHED

    def update(self, **changes) -> RLMState:
        """Return a new state with the given fields changed."""
        return self.model_copy(update=changes)
