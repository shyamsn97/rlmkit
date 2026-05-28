"""Per-agent transcript persistence.

Maintains ``session/<aid>/transcript.json`` — the flat LLM I/O record
for each agent, growing turn-by-turn as the run progresses.

Kept out of :mod:`rlmflow.rlm` so the engine itself stays focused on
the state machine. The recorder is a thin wrapper over the active
:class:`~rlmflow.workspace.Session`: it reads the prior
document, diffs against the new turn's input messages, and appends
only the delta.

Failures are swallowed by design — persistence must never break a run.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

from rlmflow.graph import Graph, Node
from rlmflow.llm import LLMClient, LLMUsage


class SessionLike(Protocol):
    """The slice of :class:`Session` this recorder needs."""

    def read_transcript(self, agent_id: str) -> dict[str, Any] | None: ...
    def write_transcript(self, agent_id: str, transcript: dict[str, Any]) -> None: ...


class TranscriptRecorder:
    """Append-only per-agent transcript writer.

    Each call mutates ``session/<aid>/transcript.json`` in place —
    never re-emits the full prefix. All write failures are swallowed.
    """

    def __init__(self, session: SessionLike | None) -> None:
        self.session = session

    @property
    def enabled(self) -> bool:
        s = self.session
        return s is not None and hasattr(s, "write_transcript")

    def _read(self, agent_id: str) -> tuple[list[dict], list[dict]]:
        try:
            prior = self.session.read_transcript(agent_id) or {}  # type: ignore[union-attr]
        except Exception:  # pragma: no cover
            prior = {}
        return (
            list(prior.get("messages") or []),
            list(prior.get("metadata") or []),
        )

    def _write(
        self,
        agent_id: str,
        messages: list[dict],
        metadata: list[dict],
    ) -> None:
        try:
            self.session.write_transcript(  # type: ignore[union-attr]
                agent_id,
                {"agent_id": agent_id, "messages": messages, "metadata": metadata},
            )
        except Exception:  # pragma: no cover
            pass

    def record_turn(
        self,
        *,
        graph: Graph,
        last: Node,
        messages: list[dict[str, str]],
        client: LLMClient,
        force_final: bool,
        raw: str,
        usage: LLMUsage,
        elapsed_s: float,
    ) -> None:
        """Append this turn's new user nudges + the assistant reply.

        ``messages`` is the full conversation the LLM just saw; we diff
        against the prior transcript length to emit only the delta.
        """
        if not self.enabled:
            return
        prior_messages, prior_metadata = self._read(graph.agent_id)
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
        self._write(
            graph.agent_id,
            prior_messages + appended_msgs,
            prior_metadata + appended_meta,
        )

    def record_terminal(self, graph: Graph, node: Node) -> None:
        """Append the terminal observation as a trailing user message.

        ``record_turn`` only fires on LLM calls, so when an agent
        finishes via ``done(...)`` the REPL output from that final
        block never makes it into the transcript via the normal
        delta-projection path. This closes that gap so the saved
        transcript matches what the model would have seen had there
        been a next turn.
        """
        if not self.enabled:
            return
        content = getattr(node, "content", "") or ""
        if not content.strip():
            return
        prior_messages, prior_metadata = self._read(graph.agent_id)
        if prior_messages and prior_messages[-1].get("role") == "user":
            # Defensive: merge into the trailing user block rather than
            # emit two consecutive `{role: "user"}` messages
            # (Anthropic strictly rejects role-alternation violations).
            prior_messages[-1] = {
                "role": "user",
                "content": prior_messages[-1].get("content", "") + "\n\n" + content,
            }
            self._write(graph.agent_id, prior_messages, prior_metadata)
            return
        self._write(
            graph.agent_id,
            prior_messages + [{"role": "user", "content": content}],
            prior_metadata
            + [
                {
                    "ts": time.time(),
                    "terminal_node_id": node.id,
                    "terminal_seq": node.seq,
                }
            ],
        )


__all__ = ["SessionLike", "TranscriptRecorder"]
