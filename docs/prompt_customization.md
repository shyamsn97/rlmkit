# Prompt Customization

`RLMFlow` builds a system prompt from named sections. Most customization should
derive from the default builder instead of replacing the whole prompt, because
the default sections carry the REPL protocol, the
`launch_subagent` / `launch_subagents` delegation rules,
`CONTEXT`, `SESSION`, and the worked examples that keep recursive execution
well-formed.

Use full replacement only when you want to own that entire protocol yourself.

## Inspect The Prompt

Before changing the prompt, render the one your agent already sees:

```python
graph = agent.start("Summarize this document.", context=document)
print(agent.build_system_prompt(graph))
```

You can also render without starting a run:

```python
print(agent.build_system_prompt_for(
    query="Summarize this document.",
    agent_id="root",
    depth=0,
))
```

Each `Graph` stores the prompt snapshot that was used for that agent's
first call:

```python
print(graph.system_prompt)
```

## Default Builder Shape

The default builder has seven sections, in order:

| Section | Purpose |
| --- | --- |
| `role` | Opening contract + REPL namespace (`CONTEXT`, `llm_query_batched`, `launch_subagent`, `launch_subagents`, `SESSION`, `SHOW_VARS`, `print`, `done`). |
| `strategy` | When to use `llm_query_batched` vs `launch_subagent` / `launch_subagents`, "break down problems", REPL-for-computation with an inline physics example, truncation + long-context guidance. |
| `format` | REPL block fence rules + tiny inline demo. |
| `examples` | Five worked recipes (chunked scan, batched chunks, branch on delegate, program-style fanout, parallel fanout). |
| `final` | `done(...)` contract, `SHOW_VARS` reminder, closing exhortation. |
| `tools` | Runtime-generated tool list (custom user tools registered with the engine). |
| `status` | Runtime-generated agent id, depth, and config status. |

The first five render headless and back-to-back, so the rendered prompt reads
as one continuous narrative; the split exists so each piece is independently
swappable via `DEFAULT_BUILDER.update(name, ...)`. `tools` and `status` are
filled in by `RLMFlow` at build time.

## Recommended: Derive From `DEFAULT_BUILDER`

The default prompt is a `PromptBuilder`: an ordered list of named sections.
`.section(...)` returns a new builder, so the module-level default is never
mutated.

### Add Project Rules

Add a new section anywhere relative to the existing ones:

```python
from rlmflow import RLMFlow
from rlmflow.prompts.default import DEFAULT_BUILDER

project_rules = """
- Preserve API compatibility unless the task explicitly asks for a breaking change.
- Prefer small patches with focused tests.
- When changing public behavior, update docs in the same pass.
"""

prompt = DEFAULT_BUILDER.section(
    "project_rules",
    project_rules,
    title="Project Rules",
    after="final",
)

agent = RLMFlow(
    llm_client=llm,
    workspace=workspace,
    prompt_builder=prompt,
)
```

### Swap A Single Section

Replace just the piece you want to customize. The rest of the prompt is unchanged:

```python
from rlmflow.prompts.default import DEFAULT_BUILDER

domain_strategy = """
**When to delegate:** spawn one child per independent file/module. Keep the root
agent's job to planning, dispatch, and integration. Verify children mechanically
before `done()`.
"""

prompt = DEFAULT_BUILDER.update("strategy", domain_strategy)
```

### Prepend A Persona

Slip a small role section before `role` rather than overwriting the protocol:

```python
prompt = DEFAULT_BUILDER.section(
    "persona",
    "You are a recursive security auditor. Reproduce concrete risks and "
    "propose minimal fixes.",
    title="Persona",
    before="role",
)
```

### Remove A Section

You can remove sections, but only the ones you added — removing `system`
removes the entire delegation protocol.

```python
prompt = DEFAULT_BUILDER.remove("project_rules")
```

### Build A Prompt From Scratch

Use this when you want complete control while still using the section renderer.
If you include sections named `tools` or `status`, `RLMFlow` fills them at
runtime.

```python
from rlmflow.prompts import PromptBuilder

prompt = (
    PromptBuilder()
    .section("role", "You are a minimal REPL agent.", title="Role")
    .section(
        "protocol",
        """
- Use exactly one ```repl``` block per assistant message.
- Call `done(answer)` exactly once when finished.
- Use tools to inspect or modify files.
""",
        title="Protocol",
    )
    .section("tools", title="Tools")
    .section("status", title="Status")
)
```

## Full System Prompt Replacement

`RLMConfig.system_prompt` bypasses the builder entirely:

```python
from rlmflow import RLMConfig, RLMFlow

agent = RLMFlow(
    llm_client=llm,
    workspace=workspace,
    config=RLMConfig(
        system_prompt="""
You are a Python REPL agent.

- Use exactly one ```repl``` block per assistant message.
- Use available tools to make progress.
- Call `done(answer)` exactly once when finished.
""",
    ),
)
```

This is the most fragile option. If the prompt omits `launch_subagent`,
`launch_subagents`, `CONTEXT`, `SESSION`, or the `done(...)` rule, the model
will not reliably use those features.

## Dynamic Prompts

Subclass `RLMFlow` when the prompt should depend on the current agent,
depth, query, available tools, or project state. The hook receives the
agent's `Graph` — all run-invariants are flat fields on it
(`agent_id`, `depth`, `query`, `config`, `model`, …).

```python
from rlmflow import RLMFlow
from rlmflow.graph import Graph
from rlmflow.prompts.default import DEFAULT_BUILDER


class AuditFlow(RLMFlow):
    def build_system_prompt(self, graph: Graph) -> str:
        extra = (
            "At root depth, produce an executive summary after verification."
            if graph.depth == 0
            else "As a child call, return only structured findings."
        )

        builder = DEFAULT_BUILDER.section(
            "audit_depth_rules",
            extra,
            title="Depth Rules",
            after="strategy",
        )
        return builder.build(
            tools=self.build_tools_section(),
            status=self.build_status_section(graph),
        )
```

You can also override narrower hooks:

```python
class MyFlow(RLMFlow):
    def build_tools_section(self) -> str:
        tools = super().build_tools_section()
        return tools + "\n- Prefer read-only tools before write tools."

    def build_messages(self, graph, *, force_final=False):
        messages = super().build_messages(graph, force_final=force_final)
        # Add or transform chat messages here.
        return messages
```

## Child-Specific Prompts

The easiest way to steer a child is the query you pass to `launch_subagent` /
`launch_subagents`. Use the global prompt for stable behavior and use child
queries for local contracts.

```python
results = await launch_subagents([
    {
        "name": "api",
        "query": "Implement src/api.py. Return ONLY JSON {\"files\": [str], \"checks\": [str]}.",
        "context": api_spec,
    },
    {
        "name": "tests",
        "query": "Implement tests for src/api.py. Return ONLY JSON {\"files\": [str], \"checks\": [str]}.",
        "context": test_spec,
    },
])
```

If every child of a flow needs a different system prompt, use a subclass and
branch on `graph.depth`, `graph.agent_id`, or `graph.query`.
