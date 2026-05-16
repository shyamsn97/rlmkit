"""Building LLM messages and the system prompt.

Two responsibilities:

1. :func:`build_messages` — render an agent's trajectory as a list
   of ``{role, content}`` dicts for the LLM.
2. :func:`build_system_prompt` (and helpers) — render the system
   prompt from the configured prompt builder, including the tools
   section and the depth/status section.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rlmflow.engine.seq import iteration_count
from rlmflow.graph import (
    Graph,
    is_errored,
    is_exec_output,
    is_llm_output,
    is_user_query,
)
from rlmflow.prompts.messages import (
    CONTEXT_HINT_ABSENT,
    CONTEXT_HINT_PRESENT,
    CONTINUE_ACTION,
    FINAL_ANSWER_ACTION,
    STATUS_DEPTH_MID,
    STATUS_DEPTH_NEAR_MAX,
    STATUS_DEPTH_ROOT,
    TRUNCATION_SESSION_HINT,
    TRUNCATION_SUMMARY,
)

if TYPE_CHECKING:
    from rlmflow.rlm import RLMFlow


def build_messages(
    engine: "RLMFlow",
    graph: Graph,
    *,
    force_final: bool = False,
) -> list[dict[str, str]]:
    """Render ``graph``'s trajectory as a chat-message list for the LLM."""
    system_content = graph.system_prompt or build_system_prompt_for(
        engine,
        query=graph.query,
        agent_id=graph.agent_id,
        depth=graph.depth,
    )
    system = {"role": "system", "content": system_content}

    try:
        payload = engine.context.read("context", agent_id=graph.agent_id)
    except KeyError:
        payload = ""
    context_hint = CONTEXT_HINT_PRESENT if payload else CONTEXT_HINT_ABSENT

    msgs: list[dict[str, str]] = []
    for state in graph.states:
        if is_user_query(state):
            msgs.append({"role": "user", "content": state.content})
        elif is_llm_output(state):
            msgs.append({"role": "assistant", "content": state.reply})
        elif is_exec_output(state):
            msgs.append({"role": "user", "content": state.content or state.output})
        elif is_errored(state):
            msgs.append({"role": "user", "content": state.content})
        # SupervisingOutput, DoneOutput, and every ActionNode are
        # engine bookkeeping — not part of the LLM projection.

    cap = engine.config.max_messages
    if cap and len(msgs) > cap:
        msgs = [
            {
                "role": "user",
                "content": TRUNCATION_SUMMARY.format(
                    query=graph.query,
                    total=len(msgs),
                    cap=cap,
                    session_hint=TRUNCATION_SESSION_HINT,
                ),
            }
        ] + msgs[-cap:]

    if force_final:
        msgs.append({"role": "user", "content": FINAL_ANSWER_ACTION})
    elif iteration_count(graph) > 0:
        msgs.append(
            {
                "role": "user",
                "content": CONTINUE_ACTION.format(
                    query=graph.query, context_hint=context_hint
                ),
            }
        )
    return [system] + msgs


# ── system prompt ─────────────────────────────────────────────────────


def build_system_prompt_for(
    engine: "RLMFlow",
    *,
    query: str,
    agent_id: str,
    depth: int,
    config: dict[str, Any] | None = None,
) -> str:
    stub = Graph(
        agent_id=agent_id,
        depth=depth,
        query=query,
        config=config or node_config(engine),
    )
    return build_system_prompt(engine, stub)


def build_system_prompt(engine: "RLMFlow", graph: Graph) -> str:
    """Render the system prompt for the agent rooted in ``graph``."""
    if engine.config.system_prompt:
        return engine.config.system_prompt
    return engine.prompt_builder.build(
        tools=build_tools_section(engine),
        status=build_status_section(engine, graph),
    )


def build_tools_section(engine: "RLMFlow") -> str:
    baseline = engine.config.max_depth == 0
    tool_defs = engine.runtime.get_tool_defs()
    if baseline:
        tool_defs = [t for t in tool_defs if t.name not in ("delegate", "wait")]
    lines = [
        f"- `{tool_def.name}{tool_def.signature}`: {tool_def.description}"
        for tool_def in tool_defs
    ]
    if len(engine.llm_clients) > 1 and not baseline:
        lines.append("\nAvailable models for `delegate(model=...)`:")
        for key in sorted(engine.llm_clients):
            desc = engine.model_descriptions.get(key)
            lines.append(f"- `{key}`: {desc}" if desc else f"- `{key}`")
    modules = engine.runtime.available_modules()
    if modules:
        lines.append(f"\nPre-imported: `{'`, `'.join(modules)}`")
    return "\n".join(lines)


def build_status_section(engine: "RLMFlow", graph: Graph) -> str:
    max_depth = graph.config.get("max_depth", engine.config.max_depth)
    if max_depth == 0:
        return (
            "Baseline mode: no sub-agents available. Do all work directly "
            "in this REPL."
        )
    note = f"You are at recursion depth **{graph.depth}** of max **{max_depth}**."
    if graph.depth == 0:
        note += STATUS_DEPTH_ROOT
    elif graph.depth >= max_depth - 1:
        note += STATUS_DEPTH_NEAR_MAX
    elif graph.depth > 0:
        note += STATUS_DEPTH_MID
    return note


def node_config(engine: "RLMFlow") -> dict[str, Any]:
    """The default config dict written onto every fresh :class:`Graph`."""
    return {
        "model": "default",
        "max_depth": engine.config.max_depth,
        "max_iterations": engine.config.max_iterations,
        "max_output_length": engine.config.max_output_length,
        "max_messages": engine.config.max_messages,
        "child_max_iterations": engine.config.child_max_iterations,
        "single_block": engine.config.single_block,
        "max_budget": engine.config.max_budget,
    }


__all__ = [
    "build_messages",
    "build_status_section",
    "build_system_prompt",
    "build_system_prompt_for",
    "build_tools_section",
    "node_config",
]
