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

For the runtime contract around `yield wait(...)` and `ResumeAction`,
the full REPL/yield protocol, persistence layout, and engine
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
from rlmflow.prompts.default import DEFAULT_BUILDER, GUARDRAILS_TEXT

agent = RLMFlow(..., prompt_builder=(
    DEFAULT_BUILDER
    .section("role", "You are a security auditor.", title="Role")
    .section("guardrails", GUARDRAILS_TEXT, title="Guardrails", after="recursion")
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
sample  = CONTEXT.read(0, 2000) # char slice
window  = CONTEXT.lines(0, 50)  # line slice
hits    = CONTEXT.grep(r"TODO") # lineno:line rows
full    = CONTEXT.read()        # full payload for handoff to a child
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
- Pass `CONTEXT.read()` only when the child genuinely needs the
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
  walkthrough of stepping, workspace persistence, session reads, time travel,
  and gym-style stepping.
- [`examples/notebooks/coding_agent.ipynb`](../examples/notebooks/coding_agent.ipynb)
  — live LLM run that produces a real workspace.
- [`examples/notebooks/node_basics.ipynb`](../examples/notebooks/node_basics.ipynb)
  — querying the `Graph` API on the deterministic fixture.
- [`examples/notebooks/viz_walkthrough.ipynb`](../examples/notebooks/viz_walkthrough.ipynb)
  — every visualization helper against the same fixture.
