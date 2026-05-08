# Control

`step(node) -> node'` is the core transition. Nodes are immutable, so
checkpoint, rewind, and intervention are explicit graph operations.

## Step loop

```python
node = agent.start(query)
while not node.finished:
    node = agent.step(node)
```

`agent.run(query)` does the same thing and returns `node.get_result()`.
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
`CONTEXT`. The root agent's payload is keyword-only and optional:

```python
node = agent.start("answer from the payload", context=large_text)
```

Inside the REPL, agents see `CONTEXT` (read-only payload), `SESSION`
(read-only view of every other agent in the run), and the standard
filesystem tools. Sample, slice, or fork:

```python
CONTEXT.info()                  # {"chars": int, "lines": int}
sample  = CONTEXT.read(0, 2000) # char slice
window  = CONTEXT.lines(0, 50)  # line slice
hits    = CONTEXT.grep(r"TODO") # lineno:line rows
snap    = CONTEXT.fork()        # snapshot for handoff to a child
```

## Delegation

Children are spawned with a positional, mandatory `context`:

```python
delegate(name, query, context, *, model=None)
```

- Pass `""` when the child works from the query alone (the most common
  case for code-only tasks).
- Pass a `CONTEXT.lines(...)` / `CONTEXT.read(...)` slice when each
  child reasons over a different chunk of the parent's payload
  (chunk-and-aggregate).
- Pass `CONTEXT.fork()` only when the child genuinely needs the
  parent's full view (reviewers, auditors, deterministic retry).

The default prompt biases toward **inline first**: if you (the parent)
already know how to produce the answer end-to-end — a known multi-file
artifact, a familiar algorithm, a self-contained transform — write it
yourself with `write_file` / direct compute. Reserve `delegate(...)`
for work that is both parallel **and** requires distinct reasoning per
child (different data slices, different sources, separate verification).
Cross-file schema drift between siblings is the #1 multi-file failure
mode; inlining sidesteps it entirely.

## Walkthroughs

- [`examples/showcase.py`](../examples/showcase.py) — runnable
  walkthrough of stepping, checkpointing, intervention, and rewind.
- [`examples/notebooks/coding_agent.ipynb`](../examples/notebooks/coding_agent.ipynb)
  — generates the canonical trace under
  `examples/data/notebook-coding-agent/`.
- [`examples/notebooks/node_basics.ipynb`](../examples/notebooks/node_basics.ipynb)
  — querying that trace via the typed node API.
- [`examples/notebooks/viz_walkthrough.ipynb`](../examples/notebooks/viz_walkthrough.ipynb)
  — every visualization helper against the same trace.
