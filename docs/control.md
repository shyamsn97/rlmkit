# Control

`step(node) -> node'` is the core transition. Nodes are immutable, so
checkpoint, rewind, and intervention are explicit graph operations.

## Step loop

```python
node = agent.start(query)
while not node.finished:
    node = agent.step(node)
```

`agent.run(query)` does the same thing and returns `node.result`.
`agent.chat(messages)` is the `LLMClient` interface — same loop, last
user message becomes the query.

## Checkpoint / resume

```python
from rlmflow import Node

node.save("ckpt.json")

node = Node.load("ckpt.json")
while not node.finished:
    node = agent.step(node)
```

## Rewind

Keep every node snapshot in a list and resume any one of them:

```python
history = [agent.start(query)]
while not history[-1].finished:
    history.append(agent.step(history[-1]))

node = history[-5]
while not node.finished:
    node = agent.step(node)
```

## Branch Workspaces

Use `Workspace.fork(...)` when a branch needs isolated files, session, and
context stores:

```python
branch = workspace.fork(new_branch_id="repair", new_dir="./repair-workspace")
```

## Intervene

Between steps, patch node fields with `node.update(**changes)` or replace
subtrees with `node.replace_many(...)`:

```python
node = node.update(
    children=[
        child for child in node.children
        if getattr(child, "agent_id", "") != "root.bad_branch"
    ],
)
```

Use this to remove runaway children, adjust config, request termination, or
replace a leaf with a manually constructed `ResultNode`.

## Custom runtime

Subclass `Runtime` and implement two methods:

```python
class MyRuntime(Runtime):
    def send(self, msg: dict) -> None: ...
    def recv(self) -> dict: ...
```

See [`runtimes.md`](runtimes.md).

## Custom tools

```python
@runtime.tool("Search files for a regex.")
def search(pattern: str, path: str = ".") -> str:
    ...
```

Or pass a list to `runtime.register_tools([...])`.

## Custom prompt

```python
from rlmflow.prompts.default import DEFAULT_BUILDER, GUARDRAILS_TEXT

agent = RLMFlow(..., prompt_builder=(
    DEFAULT_BUILDER
    .section("role", "You are a security auditor.", title="Role")
    .section("guardrails", GUARDRAILS_TEXT, title="Guardrails", after="recursion")
))
```

Or subclass `RLMFlow` and override `build_system_prompt`, `build_messages`,
`extract_code`, `step_observation`, `step_action`, or `delegate_for_step`.

## Session And Context

`Workspace.session` stores the typed node/message graph:

```python
nodes = workspace.session.load()
chain = workspace.session.chain_to(node)
```

`Workspace.context` stores optional payloads exposed inside the REPL as
`CONTEXT`:

```python
node = agent.start("answer from the payload", context=large_text)
```

Inside the REPL:

```python
sample = CONTEXT.read(0, 2000)
```

See [`showcase.py`](../examples/showcase.py) for a runnable walkthrough
of everything above.
