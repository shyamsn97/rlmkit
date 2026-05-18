"""RLMFlow's data model — one recursive class.

* :class:`Graph` — one agent, mutable, with per-agent invariants as flat
  fields, ``states`` as its trajectory, and ``children`` for sub-agents.
  Recursion lives in ``children``; cross-agent navigation goes through
  ``graph[other_aid]`` or ``graph.agents``.
* :class:`AgentsView`, :class:`NodesView`, :class:`EdgesView` — flat
  query / mutation views over the subtree (``graph.agents``,
  ``graph.nodes``, ``graph.edges``).
* :class:`Node` and its subclasses — one immutable per-state payload.
  See :mod:`rlmflow.graph.node` and ``docs/internal/node_model.md``.
* :class:`WorkspaceRef`, :class:`RuntimeRef` — serializable handles to
  external systems (branch storage, durable REPL).
* :class:`ChildHandle`, :class:`WaitRequest` — REPL protocol handles
  the engine inspects for delegation / suspension.
"""

from rlmflow.graph.graph import (
    AgentsView,
    Edge,
    EdgesView,
    Graph,
    NodesView,
    RuntimeRef,
    WorkspaceRef,
)
from rlmflow.graph.handles import ChildHandle, WaitRequest
from rlmflow.graph.node import (
    ActionNode,
    CodeObservation,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    LLMAction,
    LLMOutput,
    Node,
    ObservationNode,
    ResumeAction,
    SupervisingOutput,
    UserQuery,
    is_action,
    is_code_observation,
    is_done,
    is_errored,
    is_exec_action,
    is_exec_output,
    is_llm_action,
    is_llm_output,
    is_observation,
    is_resume_action,
    is_resumed,
    is_supervising,
    is_user_query,
    new_id,
    parse_node_obj,
)
from rlmflow.graph.timeline import retrace_steps

__all__ = [
    "ActionNode",
    "AgentsView",
    "ChildHandle",
    "CodeObservation",
    "DoneOutput",
    "Edge",
    "EdgesView",
    "ErrorOutput",
    "ExecAction",
    "ExecOutput",
    "Graph",
    "LLMAction",
    "LLMOutput",
    "Node",
    "NodesView",
    "ObservationNode",
    "ResumeAction",
    "RuntimeRef",
    "SupervisingOutput",
    "UserQuery",
    "WaitRequest",
    "WorkspaceRef",
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
    "retrace_steps",
]
