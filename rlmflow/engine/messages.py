"""Building LLM messages and the system prompt.

Two responsibilities:

1. :func:`build_messages` — render an agent's trajectory as a list
   of ``{role, content}`` dicts for the LLM.
2. :func:`build_system_prompt` (and helpers) — render the system
   prompt from the configured prompt builder, including the tools
   section and the depth/status section.

Every function takes only the concrete dependencies it actually
reads (config fields, the prompt builder, the runtime root for
tool defs, etc.) — no engine class, nothing is imported from
:mod:`rlmflow.rlm`.
"""

from __future__ import annotations

from typing import Any

from rlmflow.engine.config import RLMConfig
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
from rlmflow.runtime import Runtime
from rlmflow.workspace import Context


def build_messages(
    graph: Graph,
    *,
    force_final: bool = False,
    context: Context,
    config: RLMConfig,
    runtime: Runtime,
    llm_clients: dict[str, Any],
    model_descriptions: dict[str, str],
    prompt_builder: Any,
) -> list[dict[str, str]]:
    """Render ``graph``'s trajectory as a chat-message list for the LLM."""
    system_content = graph.system_prompt or build_system_prompt_for(
        query=graph.query,
        agent_id=graph.agent_id,
        depth=graph.depth,
        config=config,
        runtime=runtime,
        llm_clients=llm_clients,
        model_descriptions=model_descriptions,
        prompt_builder=prompt_builder,
    )
    system = {"role": "system", "content": system_content}

    try:
        payload = context.read("context", agent_id=graph.agent_id)
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

    cap = config.max_messages
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

    # Gate on LLMOutput count — not LLMAction count — so we don't
    # double up the user prompt on the very first turn. The
    # transition writes the paired ``LLMAction`` *before* calling
    # ``build_messages``, so the action for the in-progress turn is
    # already in ``graph.states`` here. ``LLMOutput``s only exist
    # for *completed* prior turns, which is what "should we nudge
    # with CONTINUE_ACTION?" actually wants to know.
    has_prior_turn = any(is_llm_output(s) for s in graph.states)
    if force_final:
        msgs.append({"role": "user", "content": FINAL_ANSWER_ACTION})
    elif has_prior_turn:
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
    *,
    query: str,
    agent_id: str,
    depth: int,
    config: RLMConfig,
    runtime: Runtime,
    llm_clients: dict[str, Any],
    model_descriptions: dict[str, str],
    prompt_builder: Any,
    sub_config: dict[str, Any] | None = None,
) -> str:
    """Render the system prompt for a (possibly not-yet-instantiated) agent."""
    stub = Graph(
        agent_id=agent_id,
        depth=depth,
        query=query,
        config=sub_config or node_config(config),
    )
    return build_system_prompt(
        stub,
        config=config,
        runtime=runtime,
        llm_clients=llm_clients,
        model_descriptions=model_descriptions,
        prompt_builder=prompt_builder,
    )


def build_system_prompt(
    graph: Graph,
    *,
    config: RLMConfig,
    runtime: Runtime,
    llm_clients: dict[str, Any],
    model_descriptions: dict[str, str],
    prompt_builder: Any,
) -> str:
    """Render the system prompt for the agent rooted in ``graph``."""
    if config.system_prompt:
        return config.system_prompt
    return prompt_builder.build(
        tools=build_tools_section(
            runtime=runtime,
            max_depth=config.max_depth,
            llm_clients=llm_clients,
            model_descriptions=model_descriptions,
        ),
        status=build_status_section(graph, max_depth=config.max_depth),
    )


def build_tools_section(
    *,
    runtime: Runtime,
    max_depth: int,
    llm_clients: dict[str, Any],
    model_descriptions: dict[str, str],
) -> str:
    baseline = max_depth == 0
    tool_defs = runtime.get_tool_defs()
    if baseline:
        tool_defs = [t for t in tool_defs if t.name not in ("delegate", "wait")]
    lines = [
        f"- `{tool_def.name}{tool_def.signature}`: {tool_def.description}"
        for tool_def in tool_defs
    ]
    if len(llm_clients) > 1 and not baseline:
        lines.append("\nAvailable models for `delegate(model=...)`:")
        for key in sorted(llm_clients):
            desc = model_descriptions.get(key)
            lines.append(f"- `{key}`: {desc}" if desc else f"- `{key}`")
    modules = runtime.available_modules()
    if modules:
        lines.append(f"\nPre-imported: `{'`, `'.join(modules)}`")
    return "\n".join(lines)


def build_status_section(graph: Graph, *, max_depth: int) -> str:
    effective_max = graph.config.get("max_depth", max_depth)
    if effective_max == 0:
        return (
            "Baseline mode: no sub-agents available. Do all work directly "
            "in this REPL."
        )
    note = f"You are at recursion depth **{graph.depth}** of max **{effective_max}**."
    if graph.depth == 0:
        note += STATUS_DEPTH_ROOT
    elif graph.depth >= effective_max - 1:
        note += STATUS_DEPTH_NEAR_MAX
    elif graph.depth > 0:
        note += STATUS_DEPTH_MID
    return note


def node_config(config: RLMConfig) -> dict[str, Any]:
    """The default config dict written onto every fresh :class:`Graph`."""
    return {
        "model": "default",
        "max_depth": config.max_depth,
        "max_iterations": config.max_iterations,
        "max_output_length": config.max_output_length,
        "max_messages": config.max_messages,
        "child_max_iterations": config.child_max_iterations,
        "single_block": config.single_block,
        "max_budget": config.max_budget,
    }


__all__ = [
    "build_messages",
    "build_status_section",
    "build_system_prompt",
    "build_system_prompt_for",
    "build_tools_section",
    "node_config",
]
