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
    """Opaque reference returned by delegate(), passed to wait_all()."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id

    def __repr__(self) -> str:
        return f"ChildHandle({self.agent_id!r})"


# ── RLMState ─────────────────────────────────────────────────────────


class RLMState(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent_id: str
    status: Status
    iteration: int = 0
    config: dict = {}

    event: StepEvent | None = None
    messages: list[dict] = []
    last_reply: str | None = None
    result: str | None = None
    context: str | None = None

    children: list[RLMState] = []

    @property
    def finished(self) -> bool:
        return self.status == Status.FINISHED

    def update(self, **changes) -> RLMState:
        """Return a new state with the given fields changed."""
        return self.model_copy(update=changes)
