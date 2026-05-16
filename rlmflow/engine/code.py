"""Call the LLM, extract a code block.

This module wraps the LLM-side of one engine step:

- :func:`reply_to` — call the LLM for one turn and produce the
  resulting :class:`~rlmflow.graph.LLMOutput`.
- :func:`call_llm` and :func:`extract_code` — small helpers used by
  :func:`reply_to`.

The action-bookkeeping side (writing the paired :class:`LLMAction`
*before* the call, plus the subsequent :class:`ExecAction` and
:class:`CodeObservation`) lives in :mod:`rlmflow.engine.transitions`.
Keeping LLM I/O here keeps the LLM-shaped helpers in one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rlmflow.graph import Graph, LLMOutput, Node
from rlmflow.llm import LLMClient, LLMUsage
from rlmflow.utils import find_code_blocks

if TYPE_CHECKING:
    from rlmflow.rlm import RLMFlow


def reply_to(
    engine: "RLMFlow",
    graph: Graph,
    last: Node,
    *,
    force_final: bool,
) -> LLMOutput:
    """Ask the LLM for the next turn and return an :class:`LLMOutput`.

    Always returns an :class:`LLMOutput`, even when the reply has no
    parseable code block (in which case ``LLMOutput.code`` is ``""``).
    The caller is responsible for handling the empty-code case by
    appending a follow-up :class:`ErrorOutput` (with
    ``error="no_code_block"``) — that way the trajectory keeps a
    faithful record that the LLM was called and replied, and the
    engine's "retry, no code block" message becomes its own node
    rather than a synthetic substitute.
    """
    from rlmflow.engine.messages import build_messages

    messages = build_messages(engine, graph, force_final=force_final)
    client = llm_client_for(engine, graph)
    raw = call_llm(engine, messages, client=client)
    usage = client.last_usage or LLMUsage()
    code = extract_code(engine, raw)
    return LLMOutput(
        agent_id=graph.agent_id,
        seq=last.seq + 1,
        reply=raw,
        code=code or "",
        model=getattr(client, "model", None),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )


def call_llm(
    engine: "RLMFlow",
    messages: list[dict[str, str]],
    *,
    client: LLMClient | None = None,
) -> str:
    active_client = client or engine.llm_client
    result = "".join(active_client.stream(messages))
    engine.last_usage = active_client.last_usage
    return result


def llm_client_for(engine: "RLMFlow", graph: Graph) -> LLMClient:
    model = graph.config.get("model", "default")
    return engine.llm_clients.get(model, engine.llm_client)


def extract_code(engine: "RLMFlow", text: str) -> str | None:
    blocks = find_code_blocks(text)
    if not blocks:
        return None
    return blocks[0] if engine.config.single_block else "\n\n".join(blocks)


__all__ = [
    "call_llm",
    "extract_code",
    "llm_client_for",
    "reply_to",
]
