"""Typed agent states — the obs / action trajectory taxonomy.

Every trajectory is a strictly alternating chain of *observations*
(input the system received) and *actions* (work the system did in
response). Every action is followed by exactly one observation.

Hierarchy::

    Node
    ├── ObservationNode               base — inputs the system received
    │   ├── UserQuery                   bootstrap input
    │   ├── LLMOutput                   what the LLM returned
    │   └── CodeObservation             base — anything from running code
    │       ├── ExecOutput                normal stdout
    │       ├── SupervisingOutput        code yielded, scheduler is waiting
    │       ├── ErrorOutput               code errored
    │       └── DoneOutput                code called done(); terminal
    └── ActionNode                    base — work the system did
        ├── LLMAction                   called the LLM
        ├── ExecAction                  ran the LLM's fresh code
        └── ResumeAction                supervisor resumed paused code

See ``docs/internal/node_model.md`` for the full spec.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


def new_id() -> str:
    return f"n_{uuid4().hex[:12]}"


# ── base ─────────────────────────────────────────────────────────────


class Node(BaseModel):
    """One immutable state in an agent's trajectory.

    Concrete subclasses live under :class:`ObservationNode` and
    :class:`ActionNode`. Agent-invariant data lives directly on
    :class:`~rlmflow.graph.Graph`; cross-agent topology is recovered
    from the recursive ``Graph.children`` structure (no separate
    edge objects are stored).
    """

    model_config = ConfigDict(frozen=True)

    type: str
    id: str = Field(default_factory=new_id)
    agent_id: str = "root"
    seq: int = 0

    @property
    def terminal(self) -> bool:
        return False

    def update(self, **changes: Any) -> Node:
        return self.model_copy(update=changes)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


# ── intermediate bases ───────────────────────────────────────────────


class ObservationNode(Node):
    """Base for nodes that record an *input the system received*.

    Subtypes: :class:`UserQuery`, :class:`LLMOutput`,
    :class:`CodeObservation` (and its four subtypes).
    """


class ActionNode(Node):
    """Base for nodes that record *work the system did*.

    Subtypes: :class:`LLMAction`, :class:`ExecAction`,
    :class:`ResumeAction`.
    """


class CodeObservation(ObservationNode):
    """Base for any observation that came out of running code.

    The four leaf subtypes — :class:`ExecOutput`,
    :class:`SupervisingOutput`, :class:`ErrorOutput`,
    :class:`DoneOutput` — share the ``resumed_from`` field (empty
    when produced by an :class:`ExecAction`, populated when
    produced by a :class:`ResumeAction`).
    """

    resumed_from: list[str] = Field(default_factory=list)


# ── observations ─────────────────────────────────────────────────────


class UserQuery(ObservationNode):
    """The bootstrap input the agent received.

    Always at ``seq=0``. ``content`` is the user-visible "first
    user-role message" rendered into the LLM's first turn — the
    raw user query for root agents, or the formatted spawn prompt
    for child agents.
    """

    type: Literal["user_query"] = "user_query"
    content: str = ""


class LLMOutput(ObservationNode):
    """What the LLM returned for one turn.

    ``reply`` is the raw model output. ``code`` is the extracted
    code block (or ``""`` if the reply had no fence — that flake
    is recorded faithfully here, then followed by an
    :class:`ErrorOutput` with ``error="no_code_block"``).
    """

    type: Literal["llm_output"] = "llm_output"
    reply: str = ""
    code: str = ""
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class ExecOutput(CodeObservation):
    """The runtime executed code and produced normal stdout output.

    ``output`` is the raw captured stdout. ``content`` is the same
    text wrapped for the LLM's next user-role message (the
    "REPL output: ..." framing).
    """

    type: Literal["exec_output"] = "exec_output"
    output: str = ""
    content: str = ""


class SupervisingOutput(CodeObservation):
    """The runtime paused at a ``yield rlm_wait(...)``.

    The parent agent's runtime is suspended and the scheduler is
    gating on the children listed in ``waiting_on``. Once they all
    settle (success or failure), the engine emits a
    :class:`ResumeAction` and the next observation reflects the
    resumed continuation.

    ``output`` is whatever the code printed *before* yielding (may
    be empty). The post-resume output appears on the next
    :class:`CodeObservation`.
    """

    type: Literal["supervising_output"] = "supervising_output"
    output: str = ""
    waiting_on: list[str] = Field(default_factory=list)


class ErrorOutput(CodeObservation):
    """The code execution errored.

    ``error`` is the error kind (``"no_code_block"``,
    ``"invalid_yield"``, ``"orphaned_delegates"``,
    ``"exec_exception"``, ``"max_iterations"``, …). ``content`` is the
    user-visible retry message rendered into the next LLM turn.
    """

    type: Literal["error_output"] = "error_output"
    error: str = ""
    content: str = ""
    output: str = ""


class DoneOutput(CodeObservation):
    """The code called ``done(...)``. Terminal.

    ``result`` is the final answer the agent produced. ``content``
    is the rendered "agent done" message used when this child's
    result is surfaced to a parent's resume.
    """

    type: Literal["done_output"] = "done_output"
    result: str = ""
    content: str = ""
    output: str = ""

    @property
    def terminal(self) -> bool:
        return True


# ── actions ──────────────────────────────────────────────────────────


class LLMAction(ActionNode):
    """The engine called the LLM.

    Carries call-time metadata only. The reply / code lives on the
    paired :class:`LLMOutput` that follows.
    """

    type: Literal["llm_action"] = "llm_action"
    model: str | None = None


class ExecAction(ActionNode):
    """The engine ran the LLM's fresh code.

    ``code`` is an optional echo of what we ran; the source of
    truth is the preceding :class:`LLMOutput`'s ``code``. The
    result of the run lives on the paired :class:`CodeObservation`
    that follows.
    """

    type: Literal["exec_action"] = "exec_action"
    code: str = ""


class ResumeAction(ActionNode):
    """The supervisor resumed paused code.

    Distinct from :class:`ExecAction` because it represents a
    different system decision: the supervisor (not the LLM) chose
    to drive the runtime forward. ``resumed_from`` lists the child
    agent ids whose completions cleared the wait. The result of
    the resume lives on the paired :class:`CodeObservation` that
    follows (any of the four ``CodeObservation`` subtypes).
    """

    type: Literal["resume_action"] = "resume_action"
    code: str = ""
    resumed_from: list[str] = Field(default_factory=list)


# ── predicates ───────────────────────────────────────────────────────


def is_observation(node: Node) -> bool:
    return isinstance(node, ObservationNode)


def is_action(node: Node) -> bool:
    return isinstance(node, ActionNode)


def is_code_observation(node: Node) -> bool:
    return isinstance(node, CodeObservation)


def is_user_query(node: Node) -> bool:
    return isinstance(node, UserQuery)


def is_llm_output(node: Node) -> bool:
    return isinstance(node, LLMOutput)


def is_exec_output(node: Node) -> bool:
    return isinstance(node, ExecOutput)


def is_supervising(node: Node) -> bool:
    return isinstance(node, SupervisingOutput)


def is_errored(node: Node) -> bool:
    return isinstance(node, ErrorOutput)


def is_done(node: Node) -> bool:
    return isinstance(node, DoneOutput)


def is_llm_action(node: Node) -> bool:
    return isinstance(node, LLMAction)


def is_exec_action(node: Node) -> bool:
    return isinstance(node, ExecAction)


def is_resume_action(node: Node) -> bool:
    return isinstance(node, ResumeAction)


def is_resumed(node: Node) -> bool:
    """A :class:`CodeObservation` that came from a :class:`ResumeAction`."""
    return isinstance(node, CodeObservation) and bool(node.resumed_from)


# ── parser ───────────────────────────────────────────────────────────


NodeUnion = Annotated[
    Union[
        UserQuery,
        LLMOutput,
        ExecOutput,
        SupervisingOutput,
        ErrorOutput,
        DoneOutput,
        LLMAction,
        ExecAction,
        ResumeAction,
    ],
    Field(discriminator="type"),
]


_NODE_ADAPTER: TypeAdapter[Node] = TypeAdapter(NodeUnion)


def parse_node_obj(data: dict) -> Node:
    return _NODE_ADAPTER.validate_python(data)


__all__ = [
    "ActionNode",
    "CodeObservation",
    "DoneOutput",
    "ErrorOutput",
    "ExecAction",
    "ExecOutput",
    "LLMAction",
    "LLMOutput",
    "Node",
    "ObservationNode",
    "ResumeAction",
    "SupervisingOutput",
    "UserQuery",
    "is_action",
    "is_code_observation",
    "is_done",
    "is_errored",
    "is_exec_action",
    "is_exec_output",
    "is_llm_action",
    "is_llm_output",
    "is_observation",
    "is_resume_action",
    "is_resumed",
    "is_supervising",
    "is_user_query",
    "new_id",
    "parse_node_obj",
]
