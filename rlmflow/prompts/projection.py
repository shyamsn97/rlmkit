"""Helpers for projecting graph states into chat messages."""

from __future__ import annotations

from rlmflow.graph.node import (
    Node,
    is_errored,
    is_exec_output,
    is_llm_output,
    is_user_query,
)


def append_message(msgs: list[dict[str, str]], role: str, content: str) -> None:
    if msgs and role == "user" and msgs[-1]["role"] == "user":
        msgs[-1] = {
            "role": "user",
            "content": msgs[-1]["content"] + "\n\n" + content,
        }
        return
    msgs.append({"role": role, "content": content})


def coalesce_messages(msgs: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for msg in msgs:
        append_message(out, msg["role"], msg["content"])
    return out


def project_state_messages(
    states: list[Node],
    *,
    skip_empty: bool = False,
) -> list[dict[str, str]]:
    """Project persisted states to LLM-style messages.

    Action nodes, supervising observations, and terminal observations are engine
    bookkeeping and intentionally omitted from this projection.
    """

    msgs: list[dict[str, str]] = []
    for state in states:
        role: str | None = None
        content = ""
        if is_user_query(state):
            role = "user"
            content = state.content
        elif is_llm_output(state):
            role = "assistant"
            content = state.reply
        elif is_exec_output(state):
            role = "user"
            content = state.content or state.output
        elif is_errored(state):
            role = "user"
            content = state.content

        if role is None:
            continue
        if skip_empty and not content:
            continue
        append_message(msgs, role, content)
    return msgs


__all__ = ["append_message", "coalesce_messages", "project_state_messages"]
