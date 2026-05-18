"""Call the LLM, extract a code block.

This module wraps the LLM-side of one engine step:

- :func:`reply_to` — call the LLM for one turn and produce the
  resulting :class:`~rlmflow.graph.LLMOutput` (and the ``LLMUsage``
  the caller may want to cache).
- :func:`call_llm`, :func:`llm_client_for`, :func:`extract_code` —
  small helpers used by :func:`reply_to`. Each takes the explicit
  values it needs.

The action-bookkeeping side (writing the paired :class:`LLMAction`
*before* the call, plus the subsequent :class:`ExecAction` and
:class:`CodeObservation`) lives in :mod:`rlmflow.engine.transitions`.

No function in this module imports :class:`~rlmflow.rlm.RLMFlow`.
Every function takes its dependencies as explicit keyword arguments.
"""

from __future__ import annotations

import time
from typing import Any

from rlmflow.engine.config import RLMConfig
from rlmflow.engine.messages import build_messages
from rlmflow.graph import Graph, LLMOutput, Node
from rlmflow.llm import LLMClient, LLMUsage
from rlmflow.runtime import Runtime
from rlmflow.utils import find_code_blocks
from rlmflow.workspace import Context, Session


def reply_to(
    graph: Graph,
    last: Node,
    *,
    force_final: bool,
    session: Session,
    context: Context,
    config: RLMConfig,
    runtime: Runtime,
    llm_client: LLMClient,
    llm_clients: dict[str, Any],
    model_descriptions: dict[str, str],
    prompt_builder: Any,
) -> tuple[LLMOutput, LLMUsage]:
    """Ask the LLM for the next turn and return ``(LLMOutput, LLMUsage)``.

    Always returns an :class:`LLMOutput`, even when the reply has no
    parseable code block (in which case ``LLMOutput.code`` is ``""``).
    The caller is responsible for handling the empty-code case by
    appending a follow-up :class:`ErrorOutput` (with
    ``error="no_code_block"``).

    The returned ``LLMUsage`` is the per-call usage; the caller (e.g.
    :class:`~rlmflow.rlm.RLMFlow`) decides whether to cache it as
    ``self.last_usage``.
    """
    messages = build_messages(
        graph,
        force_final=force_final,
        context=context,
        config=config,
        runtime=runtime,
        llm_clients=llm_clients,
        model_descriptions=model_descriptions,
        prompt_builder=prompt_builder,
    )
    client = llm_client_for(graph, llm_clients=llm_clients, default=llm_client)
    t0 = time.time()
    raw, usage = call_llm(messages, client=client)
    elapsed_s = round(time.time() - t0, 3)
    code = extract_code(raw, single_block=config.single_block)
    _record_transcript(
        session=session,
        graph=graph,
        last=last,
        messages=messages,
        client=client,
        force_final=force_final,
        raw=raw,
        usage=usage,
        elapsed_s=elapsed_s,
    )
    output = LLMOutput(
        agent_id=graph.agent_id,
        seq=last.seq + 1,
        reply=raw,
        code=code or "",
        model=getattr(client, "model", None),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )
    return output, usage


def call_llm(
    messages: list[dict[str, str]],
    *,
    client: LLMClient,
) -> tuple[str, LLMUsage]:
    """Stream a chat completion from ``client`` and return ``(text, usage)``.

    Returns the post-call ``LLMUsage`` instead of mutating any
    engine-level state — the caller decides what to do with it.
    """
    text = "".join(client.stream(messages))
    usage = client.last_usage or LLMUsage()
    return text, usage


def llm_client_for(
    graph: Graph,
    *,
    llm_clients: dict[str, Any],
    default: LLMClient,
) -> LLMClient:
    """Pick the per-agent LLM client.

    The agent's ``config["model"]`` is the lookup key into
    ``llm_clients``; when missing, fall back to ``default``.
    """
    model = graph.config.get("model", "default")
    return llm_clients.get(model, default)


def extract_code(text: str, *, single_block: bool) -> str | None:
    """Pull the first (or merged) ```repl block from an LLM reply."""
    blocks = find_code_blocks(text)
    if not blocks:
        return None
    return blocks[0] if single_block else "\n\n".join(blocks)


def _record_transcript(
    *,
    session: Session | None,
    graph: Graph,
    last: Node,
    messages: list[dict[str, str]],
    client: LLMClient,
    force_final: bool,
    raw: str,
    usage: LLMUsage,
    elapsed_s: float,
) -> None:
    """Update this agent's ``transcript.json`` with the new turn.

    The transcript is a *single* document per agent that grows
    turn-by-turn — ``messages`` is the flat conversation as the
    LLM saw it across every turn so far, ``metadata`` is the
    parallel per-message list. Each call here appends only the
    *new* messages (any user nudges since the last call, plus the
    assistant reply just produced) — never the full prefix again::

        {
          "agent_id": <aid>,
          "messages": [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "..."},   # turn 1
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "..."},   # turn 2
          ],
          "metadata": [
            {}, {}, <turn1_meta>,
            {},     <turn2_meta>,
          ]
        }

    Per-assistant metadata fields: ``ts``, ``model``,
    ``force_final``, ``input_tokens``, ``output_tokens``,
    ``elapsed_s``, ``after_node_id``, ``after_seq``. Every other
    message gets ``{}``.

    Transcript-write failures are swallowed: persistence should
    never break a run.
    """
    if session is None or not hasattr(session, "write_transcript"):
        return
    try:
        prior = session.read_transcript(graph.agent_id) or {}
    except Exception:  # pragma: no cover
        prior = {}
    prior_messages: list[dict[str, str]] = list(prior.get("messages") or [])
    prior_metadata: list[dict] = list(prior.get("metadata") or [])

    # ``messages`` is the literal list passed to ``chat()``: the
    # full conversation up to (but not including) this turn's
    # assistant reply. The prior transcript ended with the
    # *previous* turn's assistant reply, so what's new this turn
    # is ``messages[len(prior_messages):]`` plus this turn's
    # assistant reply. No prefix is rewritten.
    new_inputs = messages[len(prior_messages) :]
    appended_msgs = list(new_inputs) + [{"role": "assistant", "content": raw}]
    appended_meta: list[dict] = [{} for _ in new_inputs]
    appended_meta.append(
        {
            "ts": time.time(),
            "model": getattr(client, "model", None),
            "force_final": force_final,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "elapsed_s": elapsed_s,
            "after_node_id": last.id,
            "after_seq": last.seq,
        }
    )
    transcript = {
        "agent_id": graph.agent_id,
        "messages": prior_messages + appended_msgs,
        "metadata": prior_metadata + appended_meta,
    }
    try:
        session.write_transcript(graph.agent_id, transcript)
    except Exception:  # pragma: no cover - never break runs on persistence error
        pass


__all__ = [
    "call_llm",
    "extract_code",
    "llm_client_for",
    "reply_to",
]
