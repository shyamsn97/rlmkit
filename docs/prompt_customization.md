# Prompt Customization

`RLMFlow` builds a system prompt from named sections. Most customization should
derive from the default builder instead of replacing the whole prompt, because
the default includes the REPL protocol, `delegate` / `wait` rules, `CONTEXT`,
`SESSION`, and examples that keep recursive execution well-formed.

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

## Recommended: Derive From `DEFAULT_BUILDER`

The default prompt is a `PromptBuilder`: an ordered list of named sections.
`.section(...)` returns a new builder, so the module-level default is never
mutated.

```python
from rlmflow import RLMFlow
from rlmflow.prompts.default import DEFAULT_BUILDER, ROLE_TEXT

auditor_prompt = DEFAULT_BUILDER.section(
    "role",
    "You are a recursive security auditor. Find concrete risks, reproduce them "
    "when possible, and propose minimal fixes.\n\n" + ROLE_TEXT,
    title="Role",
)

agent = RLMFlow(
    llm_client=llm,
    workspace=workspace,
    prompt_builder=auditor_prompt,
)
```

Replacing a section with the same name preserves its position. Adding a new
name appends by default, or you can place it with `before=` / `after=`.

```python
from rlmflow.prompts.default import DEFAULT_BUILDER

domain_rules = """
- Preserve API compatibility unless the task explicitly asks for a breaking change.
- Prefer small patches with focused tests.
- When changing public behavior, update docs in the same pass.
"""

prompt = DEFAULT_BUILDER.section(
    "project_rules",
    domain_rules,
    title="Project Rules",
    after="strategy",
)
```

## Common Recipes

### Replace The Role

Use this when the agent should have a different job but still follow the normal
RLMFlow execution protocol.

```python
from rlmflow.prompts.default import DEFAULT_BUILDER

prompt = DEFAULT_BUILDER.section(
    "role",
    "You are a recursive data analyst. Split independent analyses into child "
    "calls, verify aggregates mechanically, and return concise conclusions.",
    title="Role",
)
```

### Add Domain Rules

Use a new section when the default sections are right, but your project needs
extra constraints.

```python
prompt = DEFAULT_BUILDER.section(
    "domain",
    """
- All SQL must be read-only unless the user explicitly requests mutation.
- Prefer parameterized queries.
- Report row counts and any filters applied.
""",
    title="Domain Rules",
    after="guardrails",
)
```

### Tune Delegation Behavior

The `strategy`, `recursion`, and `guardrails` sections have the biggest effect
on when the agent delegates and how it waits.

```python
from rlmflow.prompts.default import DEFAULT_BUILDER, RECURSION_TEXT

prompt = (
    DEFAULT_BUILDER
    .section(
        "strategy",
        """
**Size up -> split independent work -> delegate -> verify -> done.**

1. If the user asks for separate files, components, or chunks, delegate one
   child per independent part.
2. Give every child an explicit contract: filenames, signatures, output schema,
   and required checks.
3. After `yield wait(...)`, verify the combined artifact in a fresh turn before
   `done()`.
""",
        title="Strategy",
    )
    .section("recursion", RECURSION_TEXT, title="Recursion")
)
```

### Remove A Section

You can remove sections, but be careful with protocol-heavy sections like
`repl`, `recursion`, `context`, and `session`.

```python
prompt = DEFAULT_BUILDER.remove("core_examples")
```

Removing `core_examples` can make malformed delegation more likely, so prefer
replacing it with smaller examples before deleting it entirely.

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
- Respond with exactly one ```repl``` block.
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

- Respond with exactly one ```repl``` block.
- Use available tools to make progress.
- Call `done(answer)` exactly once when finished.
""",
    ),
)
```

This is the most fragile option. If the prompt omits `delegate`, `wait`,
`CONTEXT`, `SESSION`, or the `done(...)` rule, the model will not reliably use
those features.

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
        extra = ""
        if graph.depth == 0:
            extra = "At root depth, produce an executive summary after verification."
        else:
            extra = "As a child call, return only structured findings."

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

The easiest way to steer a child is the query you pass to `delegate(...)`.
Use the global prompt for stable behavior and use child queries for local
contracts.

```python
handles = [
    delegate(
        "api",
        "Implement src/api.py. Return ONLY JSON {\"files\": [str], \"checks\": [str]}.",
        api_spec,
    ),
    delegate(
        "tests",
        "Implement tests for src/api.py. Return ONLY JSON {\"files\": [str], \"checks\": [str]}.",
        test_spec,
    ),
]
results = yield wait(*handles)
```

If every child of a flow needs a different system prompt, use a subclass and
branch on `graph.depth`, `graph.agent_id`, or `graph.query`.

## Section Reference

The default builder currently uses these sections, in order:

| Section | Purpose |
| --- | --- |
| `role` | Agent identity and high-level job. |
| `repl` | Required response protocol, `done(...)`, and execution rules. |
| `strategy` | How to size up, decide, delegate, verify, and finish. |
| `tools` | Runtime-generated tool list. |
| `context` | `CONTEXT` API. |
| `recursion` | `delegate(...)`, `yield wait(...)`, and child resume behavior. |
| `session` | `SESSION` API for reading sibling/parent transcripts. |
| `guardrails` | Behavioral constraints. |
| `core_examples` | Concrete REPL patterns. |
| `status` | Runtime-generated agent id, depth, and config status. |

For most use cases, replace `role`, add one domain section, and leave `repl`,
`context`, `recursion`, `session`, and `status` intact.
