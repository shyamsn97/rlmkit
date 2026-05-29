# Control

`step(graph) -> graph'` is the core transition. `Graph` snapshots are
immutable, so stepping, rewind, and forking are explicit graph/workspace
operations.

## Step loop

```python
graph = agent.start(query)
while not graph.finished:
    graph = agent.step(graph)
```

`agent.run(query)` does the same thing and returns `graph.result()`.
`agent.chat(messages)` is the `LLMClient` interface — same loop, last
user message becomes the query.

Each `step(graph)` advances **one observation-to-observation
transition** for every agent that is ready to move. A single
"reasoning turn" of an agent (call the LLM, then run the code it
emitted) is therefore two `step()` rounds: an LLM half
(`obs → LLMAction → LLMOutput`) and an exec half
(`LLMOutput → ExecAction → CodeObservation`). This is the
finest-grained reproducible step the engine exposes — see
[`internal/node_model.md`](internal/node_model.md) for the full
state-machine spec and worked simulations.

## Eager Children

By default, children advance in synchronized `step(...)` batches. If child A's
current step takes 10 seconds and child B's current step takes 2 seconds,
child B's next step waits until child A's current step finishes.

Set `eager_children=True` when you want a work-conserving drain after a parent
awaits a launcher (`await launch_subagents([...])`):

```python
agent = RLMFlow(
    llm_client=...,
    runtime=...,
    config=RLMConfig(
        max_depth=2,
        max_iterations=30,
        max_concurrency=8,
        eager_children=True,
    ),
)
```

With that flag, children still do not run before the parent reaches the
awaited launcher. Once the parent is supervising, runnable children use
the configured pool's `run_until_idle(...)` behavior:

```text
childa.task_1 starts  # slow, 10s
childb.task_1 starts  # fast, 2s
childb.task_1 finishes
childb.task_2 starts  # starts before childa.task_1 finishes
childa.task_1 finishes
parent resumes when all waited-on children are done
```

See [`examples/eager_children.py`](../examples/eager_children.py) for a
deterministic offline demo that prints both modes side by side.

## Workspace Resume

```python
from rlmflow import Workspace

workspace = Workspace.open_path("runs/deep_research")
graph = workspace.load_graph()
while not graph.finished:
    graph = agent.step(graph)
```

The workspace session is the saved-run state.

## Rewind

Keep every `Graph` snapshot in a list and resume any one of them:

```python
history = [agent.start(query)]
while not history[-1].finished:
    history.append(agent.step(history[-1]))

graph = history[-5]
while not graph.finished:
    graph = agent.step(graph)
```

## Node Injection

Controllers can append typed nodes to a graph and commit them through the normal
step loop. This is useful for budget nudges, human feedback, and forced
finalization:

```python
from rlmflow import ExecAction, ExecOutput

graph = graph.inject(
    target="root.worker",
    node=ExecOutput(
        output="Injected controller observation: answer now.",
        content="Injected controller observation: answer now.",
    ),
    reason="message budget nearly exhausted",
)
graph = agent.step(graph)

graph = graph.inject(
    target="root.worker",
    node=ExecAction(code='done("best available answer")'),
    reason="message budget exhausted",
)
graph = agent.step(graph)
```

See [`injections.md`](injections.md) for the concise guide and
[`examples/injections.py`](../examples/injections.py) for a runnable offline demo.

## Branch workspaces

Use `Workspace.fork(...)` when a branch needs isolated files, session,
and context stores:

```python
branch = workspace.fork(new_branch_id="repair", new_dir="./repair-workspace")
```

## Replay

A persisted workspace can be rehydrated into a `Graph` and resumed:

```python
graph = workspace.session.load_graph()
while not graph.finished:
    graph = agent.step(graph)
```

The engine reads from `graph.states`, appends new states through the
session, and produces a fresh snapshot on every `step`. There is no
in-memory node graph to keep in sync with disk.

For the runtime contract around awaited launchers and `ResumeAction`,
the full REPL/await protocol, persistence layout, and engine
extension surface, see [`internals.md`](internals.md).

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

For a fuller guide, see [`prompt_customization.md`](prompt_customization.md).

```python
from rlmflow.prompts.default import DEFAULT_BUILDER

GUARDRAILS = """
- Verify before `done()`. Empty/zero/surprising results → one sanity check first.
- Ask children for structured output (JSON / list / count), parsed mechanically.
"""

agent = RLMFlow(..., prompt_builder=(
    DEFAULT_BUILDER
    .section("role", "You are a security auditor.", title="Role")
    .section("guardrails", GUARDRAILS, title="Guardrails", after="builtins")
))
```

Or subclass `RLMFlow` and override `build_system_prompt`,
`build_messages`, `extract_code`, or `step` (which is the public
`act + apply_one` entry point — see
[`internal/act_apply.md`](internal/act_apply.md)).

## Session And Context

`Workspace.session` stores the per-agent state log and the graph manifest.
The convenience `workspace.load_graph()` is the normal way to reopen the
current snapshot:

```python
graph = workspace.load_graph()
sub = graph["root.boid_js"]
print(sub.transcript())
```

On disk, the workspace keeps per-agent session logs under
`session/<agent-id>/` and a compact graph manifest at `graph.json`:

```text
workspace/
  graph.json                  # agent list + spawns edges
  session/root/agent.json
  session/root/session.jsonl
  session/root/latest.json
```

`Workspace.context` stores optional payloads exposed inside the REPL as
`CONTEXT`. The root agent's payload is keyword-only and optional:

```python
graph = agent.start("answer from the payload", context=large_text)
```

Payloads live beside the session views under `context/<agent-id>/`:

```text
workspace/
  context/root/context.txt
  context/root/context_metadata.json
```

Inside the REPL, agents see `CONTEXT` (read-only payload), `SESSION`
(read-only view of every other agent in the run), and the standard
filesystem tools. Sample, slice, or pass the full payload explicitly:

```python
CONTEXT.info()                  # {"chars": int, "lines": int}
sample  = CONTEXT.read(0, 2000) # char slice as str
window  = CONTEXT.lines(0, 50)  # line slice as list[str]
hits    = CONTEXT.grep(r"TODO") # lineno:line rows
full    = CONTEXT.read()        # full payload for handoff to a child
```

## Delegation

Agents delegate through two launchers (both must be awaited):

```python
# One child — returns its finish string.
answer = await launch_subagent(query, num_steps=None, context="", *, name="subagent", model="default")

# Many children in parallel — returns finish strings in spec order.
results = await launch_subagents([
    {"name": "a", "query": "...", "context": chunk_a},
    {"name": "b", "query": "...", "context": chunk_b},
])
```

- **Sequential** dependent steps: chain `await launch_subagent(...)` calls,
  feeding each result into the next child's `context`.
- **Parallel** independent work: pass every spec to `launch_subagents([...])`
  in one call so the engine schedules them on its pool concurrently.
- Pass `context=""` when the child works from the query alone (the most common
  case for code-only tasks).
- Pass a `CONTEXT.lines(...)` / `CONTEXT.read(...)` slice when each
  child reasons over a different chunk of the parent's payload
  (chunk-and-aggregate). A `list[str]` from `CONTEXT.lines(...)` is stored as
  newline-separated child context.
- Pass `CONTEXT.read()` only when the child genuinely needs the
  parent's full view (reviewers, auditors, deterministic retry).

(`rlm_delegate` / `rlm_wait` are the internal primitives the launchers compose
over — agents never call them directly.)

The default prompt biases toward a supervisor workflow for large context,
parallel reasoning, and split artifacts. Inline work is still fine for small,
tightly coupled tasks where delegation adds no useful ownership boundary.

## Walkthroughs

- [`examples/showcase.py`](../examples/showcase.py) — runnable
  walkthrough of stepping, workspace persistence, session reads, time travel,
  and gym-style stepping.
- [`examples/notebooks/coding_agent.ipynb`](../examples/notebooks/coding_agent.ipynb)
  — live LLM run that produces a real workspace.
- [`examples/notebooks/node_basics.ipynb`](../examples/notebooks/node_basics.ipynb)
  — querying the `Graph` API on the deterministic fixture.
- [`examples/notebooks/viz_walkthrough.ipynb`](../examples/notebooks/viz_walkthrough.ipynb)
  — every visualization helper against the same fixture.
