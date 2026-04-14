"""RLMState, StepEvent hierarchy, Status enum, and ChildHandle."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from rlmkit.session import Session

# ── Status ────────────────────────────────────────────────────────────


class Status(str, Enum):
    READY = "ready"
    EXECUTING = "executing"
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


class ResumeExec(StepEvent):
    """Generator resumed with child results."""

    output: str = ""


class ChildStep(StepEvent):
    child_events: list[StepEvent] = []
    all_done: bool = False
    exec_output: str | None = None
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
    query: str = ""
    status: Status = Status.READY
    iteration: int = 0
    config: dict = {}
    system_prompt: str | None = None

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

    @classmethod
    def from_session(
        cls,
        session: Session,
        agent_id: str = "root",
        *,
        recursive: bool = True,
        **fields,
    ) -> RLMState:
        """Reconstruct a state (or tree) by loading messages from a session store.

        Useful for resuming after a crash or migrating between backends.
        Only ``messages`` are recovered — metadata like ``status``, ``iteration``,
        and ``query`` must be supplied via *fields* or will take defaults.

        When *recursive* is True, child states are built for every agent_id in
        the session that is a descendant of *agent_id*, reconstructing the full
        tree structure.
        """
        messages = session.read(agent_id)
        if not recursive:
            return cls(agent_id=agent_id, messages=messages, **fields)

        all_ids = session.list_agents()
        prefix = agent_id + "."
        child_ids = [aid for aid in all_ids if aid.startswith(prefix)]

        direct: dict[str, list[str]] = {}
        for cid in child_ids:
            remainder = cid[len(prefix) :]
            top_part = remainder.split(".")[0]
            direct_id = f"{agent_id}.{top_part}"
            direct.setdefault(direct_id, [])

        children = [
            cls.from_session(session, did, recursive=True) for did in sorted(direct)
        ]
        return cls(
            agent_id=agent_id,
            messages=messages,
            children=children,
            **fields,
        )

    def tree(self, *, color: bool = True) -> str:
        """Render the full state tree as a string.

        >>> print(state.tree())
        root [supervising] iter 5
        ├── root.search_0 [finished] → "Found it on line 42"
        ├── root.search_1 [waiting] iter 2
        └── root.search_2 [finished] → ""
        """
        lines: list[str] = []
        _render_tree(self, lines, "", "", color)
        return "\n".join(lines)


# ── Tree rendering ───────────────────────────────────────────────────

_STATUS_COLORS = {
    "ready": "\033[34m",
    "executing": "\033[33m",
    "supervising": "\033[35m",
    "finished": "\033[32m",
}


def _node_label(state: RLMState, color: bool) -> str:
    B, D, R = ("\033[1m", "\033[2m", "\033[0m") if color else ("", "", "")
    sc = _STATUS_COLORS.get(state.status.value, "") if color else ""

    model = state.config.get("model")
    model_tag = f" {D}({model}){R}" if model else ""
    label = f"{B}{state.agent_id or 'root'}{R}{model_tag} {sc}[{state.status.value}]{R} iter {state.iteration}"
    if state.finished and state.result is not None:
        preview = state.result[:80].replace("\n", " ")
        label += f' {D}→ "{preview}"{R}'
    return label


def _render_tree(
    state: RLMState,
    out: list[str],
    prefix: str,
    child_prefix: str,
    color: bool,
) -> None:
    out.append(prefix + _node_label(state, color))
    for i, child in enumerate(state.children):
        is_last = i == len(state.children) - 1
        connector = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "
        _render_tree(
            child, out, child_prefix + connector, child_prefix + extension, color
        )
