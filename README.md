# rlmflow

<p align="center">
  <a href="https://pypi.org/project/rlmflow/"><img src="https://img.shields.io/pypi/v/rlmflow.svg?label=pypi" alt="PyPI" /></a>
  <a href="https://github.com/shyamsn97/rlmflow/pkgs/container/rlmflow"><img src="https://img.shields.io/badge/ghcr.io-rlmflow-2496ED?logo=docker&logoColor=white" alt="Docker" /></a>
</p>

A Python library for building controllable, forkable [Recursive Language Models](https://arxiv.org/abs/2512.24601).

As LLMs get better at coding, strict agent harnesses become less important.
RLMs let the model decide how to view and manipulate context, when to
delegate pieces of it to sub-agents, and how to combine the results,
all through the same clean coding interface.

**rlmflow** turns that recursive run into a live execution graph. Every
query, action, observation, child call, wait, resume, and result is a
typed state you can inspect, step, fork, replay, resume, and branch into
new workspaces. It is for people building long-context agents, recursive
coding agents, and research loops where the execution trace needs to be
as controllable as the final answer is useful. Each `start` / `step`
returns a fresh `Graph` snapshot: a recursive structure where every
`graph[id]` (node *or* agent id) returns a `Graph` rooted at that vertex.

<p align="center">
  <img src="docs/rlm_animation.gif" alt="rlmflow animation" />
</p>

## RLMs as Graphs

RLMs delegate subtasks to children, those children can delegate to their
own children, and results bubble back up. **rlmflow** represents the
whole run as one recursive type:

- **`Graph`** — one agent, frozen. Carries the agent's run-invariants
  flat on itself (`agent_id`, `depth`, `query`, `system_prompt`,
  `config`, `workspace`, `runtime`, `model`, `branch_id`,
  `parent_agent_id`, `parent_node_id`), plus its `states` trajectory
  and a `children: dict[str, Graph]` of sub-agents. Cross-agent
  navigation is `graph[other_aid]`; subtree views are `graph.agents`,
  `graph.nodes`, `graph.edges`.
- **`Node`** — one immutable state in an agent's trajectory. The
  trajectory is a strict alternation of **observations** (inputs the
  system received) and **actions** (work the system did). Nine leaf
  types under four base classes — see
  [`docs/internal/node_model.md`](docs/internal/node_model.md):
  - Observations: `UserQuery`, `LLMOutput`, `ExecOutput`,
    `SupervisingOutput`, `ErrorOutput`, `DoneOutput`.
  - Actions: `LLMAction`, `ExecAction`, `ResumeAction`.

The agent has exactly two delegation calls. `await launch_subagent(query, ...)`
runs one child and returns its answer; `await launch_subagents([...])` runs many
in parallel and returns their answers in order. An agent that delegates two
children and combines their results writes one REPL block like this:

```python
results = await launch_subagents([
    {"name": "search", "query": "Find evidence", "context": chunk_a},
    {"name": "verify", "query": "Check the answer", "context": chunk_b},
])
done(combine(results))
```

The `await` is the supervision point: it suspends the parent at a single
`WaitRequest`, the engine runs the children on its pool, then resumes the
parent with their results. The REPL supports top-level await and the engine
drives the resulting coroutine, roughly:

```python
out = coro.send(None)              # run until the await
# out is a WaitRequest([search, verify]) -> suspend the parent, run children
results = [c.result() for c in children]
coro.send(results)                 # resume; `results` is now the list
```

The REPL is stateful across blocks, so the next LLM turn can still see
`results`. The launchers must be awaited; a bare call or a top-level `yield`
are errors. (`rlm_delegate` / `rlm_wait` are the internal primitives the
launchers compose over — agents never call them directly.)

See [`docs/internals.md`](docs/internals.md) for the full protocol.

The block above becomes this execution graph (one obs/action pair
per step):

```text
UserQuery(root)
  -> LLMAction -> LLMOutput(code="await launch_subagents([search, verify])")
  -> ExecAction -> SupervisingOutput(waiting_on=[root.search, root.verify])
      -> UserQuery(root.search)  -> ... -> DoneOutput(root.search)
      -> UserQuery(root.verify)  -> ... -> DoneOutput(root.verify)
  -> ResumeAction -> ExecOutput(resumed_from=[root.search, root.verify])
  -> LLMAction -> LLMOutput(code="done(combine(...))")
  -> ExecAction -> DoneOutput(root)
```

## Install

```
pip install rlmflow               # core
pip install rlmflow[openai]       # + OpenAI client
pip install rlmflow[anthropic]    # + Anthropic client
pip install rlmflow[dspy]         # + DSPy adapter
pip install rlmflow[sandbox]      # + Modal, E2B, and Daytona runtimes
pip install rlmflow[viewer]       # + Gradio viewer (plotly)
pip install rlmflow[image]        # + static image / GIF export (kaleido)
pip install rlmflow[all]          # all of the above
```

From source:

```
git clone https://github.com/shyamsn97/rlmflow && cd rlmflow
pip install -e .
```

## Quick start

This example is all you need for a simple and interpretable recursive coding agent. see [notebook](./examples/notebooks/coding_agent.ipynb)

```python
from rlmflow import OpenAIClient, RLMConfig, RLMFlow, Workspace
from rlmflow.runtime.local import LocalRuntime
from rlmflow.tools import FILE_TOOLS
from rlmflow.utils.viewer import open_viewer

workspace = Workspace.create("./myproject")
runtime = LocalRuntime(workspace=workspace)

# Sandbox agent code inside Docker instead: drop-in replacement,
# same interface.  Build the image once with `docker build -t rlmflow:local .`
# from the repo root; see docs/runtimes.md and docs/security.md.
#
# from rlmflow.runtime.docker import DockerRuntime
# runtime = DockerRuntime("rlmflow:local", workspace=workspace)

runtime.register_tools(FILE_TOOLS)

agent = RLMFlow(
    llm_client=OpenAIClient("gpt-5"),
    runtime=runtime,
    workspace=workspace,
    config=RLMConfig(max_depth=2, max_iterations=30),
    llm_clients={ # additional llm clients to be chosen to delegate
        "fast": {
            "model": OpenAIClient("gpt-5-mini"),
            "description": "Cheap model for smaller subtasks",
        },
    },
)

query = "Build a python text-based adventure game with combat and inventory."
graph = agent.start(query)
while not graph.finished:
    graph = agent.step(graph)
    print(graph.tree())

print(graph.result())
open_viewer(workspace)
```

To let child agents drain work-conservingly after a parent reaches its
delegation wait (`await launch_subagent(...)` / `await launch_subagents(...)`),
enable `eager_children`:

```python
agent = RLMFlow(
    llm_client=OpenAIClient("gpt-5"),
    runtime=runtime,
    config=RLMConfig(
        max_depth=2,
        max_iterations=30,
        max_concurrency=8,
        eager_children=True,
    ),
)
```

With `eager_children=False`, a fast child that finishes `task_1` waits for
the rest of that parallel step before it can start `task_2`. With
`eager_children=True`, the fast child's `task_2` can start while a slow
sibling is still running `task_1`. See
[`examples/eager_children.py`](./examples/eager_children.py) for a
deterministic timestamped demo.

`Workspace.create("./myproject")` writes a debuggable workspace as it runs:
`session/<agent-id>/` holds the per-agent state log (`session.jsonl`,
`agent.json`, `latest.json`) plus `transcript.json` — the exact
turn-by-turn conversation each agent's LLM saw, with per-message metadata
(model, token counts, timing) for auditing or replay. `graph.json` is the
compact graph manifest for the whole run, and `context/<agent-id>/` holds
payloads exposed as `CONTEXT`. The workspace is the saved run: reopen it
later with `Workspace.open_path("./myproject").load_graph()` or
`open_viewer("./myproject")`.

## Drop-in `LLMClient`

`RLMFlow` implements `LLMClient`, so it is a drop-in replacement for any LLM.

```python
def ask(llm: LLMClient, q: str) -> str:
    return llm.chat([{"role": "user", "content": q}])

ask(OpenAIClient("gpt-4o-mini"), "2+2?")             # one LLM call
ask(RLMFlow(llm_client=..., runtime=...), "2+2?")    # full agent, same return type
```

Nest agents by passing one `RLMFlow` as another's `llm_client`.

## Step and inspect

`step(graph) -> graph'` is one atomic graph transition. Every step
returns a new immutable `Graph`, so the live tree is just `graph.tree()`:

```python
graph = agent.start(query)
while not graph.finished:
    graph = agent.step(graph)
print(graph.tree())
```

```text
root [supervising] {default}
├── root.scanner_auth [result] {fast} -> Found SQL injection in login.py
├── root.scanner_api  [supervising] {default}
│   ├── root.scanner_api.chunk_0 [result] {fast} -> Clean
│   └── root.scanner_api.chunk_1 [result] {fast} -> Payment flow is safe
└── root.scanner_db   [result] {fast} -> No issues found
```

Every transition follows the same obs → action → obs shape:

```text
LLMOutput  -> ExecAction -> ExecOutput          (REPL output, normal continuation)
                         -> DoneOutput          (code called done())
                         -> ErrorOutput         (code raised / no code block)
                         -> SupervisingOutput   (awaited a launcher — waiting on children)
SupervisingOutput -> ResumeAction -> ExecOutput / Done / Error / Supervising
                                                (children settled — supervisor unpaused)
ExecOutput -> LLMAction -> LLMOutput            (back to the LLM for the next turn)
```

Action nodes carry the work the engine did; observation nodes carry
what was returned. Every action is followed by exactly one
observation. The graph is queryable in plain Python:

```python
graph.tree()                                  # ASCII render
graph["root.scanner_api"]                     # sub-Graph rooted at that agent / node
graph.agents["root.scanner_api"].states       # state trajectory for one agent
graph.children                                # list[Graph] for child agents
graph.nodes.find("n_abc...")                  # bare Node lookup by id
graph.nodes.errors()                          # every ErrorOutput across agents
graph.nodes.results()                         # every DoneOutput across agents
graph.nodes.supervising()                     # every SupervisingOutput across agents
graph.nodes.where(type="llm_output", agent_id="root")  # kwargs match Node attrs
graph.nodes.where(lambda n: n.type == "exec_output")    # or pass a predicate
graph.to_dict()                               # full JSON-serializable payload
```

## Inject controller events

Because `Graph` is the control surface, external controllers can append typed
events and commit them through the normal step loop. This is useful for human
feedback, budget nudges, and forced finalization without losing traceability:

```python
from rlmflow import ExecAction, ExecOutput

graph = graph.inject(
    target="root.scanner_api",
    node=ExecOutput(
        output="Injected controller observation: answer with current evidence.",
        content="Injected controller observation: answer with current evidence.",
    ),
    reason="message budget nearly exhausted",
)
graph = agent.step(graph)  # persists the observation, then continues

graph = graph.inject(
    target="root.scanner_api",
    node=ExecAction(code='done("best available answer")'),
    reason="message budget exhausted",
)
graph = agent.step(graph)  # executes the action and writes DoneOutput
```

Injected nodes are marked in the graph (`node.injected`,
`node.injected_reason`) and persisted like normal states. See
[`docs/injections.md`](docs/injections.md) and
[`examples/injections.py`](examples/injections.py).

## Workspace, Branch, Replay

When you run with a `Workspace`, the workspace directory is the durable run:

```python
workspace = Workspace.open_path("./myproject")
graph = workspace.load_graph()
agent = RLMFlow(llm_client=AnotherModel(), workspace=workspace, ...)
while not graph.finished:
    graph = agent.step(graph)
```

To branch into an isolated workspace with its own session, context, and
working tree:

```python
alt = workspace.fork(new_branch_id="repair", new_dir="./runs/repair")
alt_agent = RLMFlow(llm_client=..., workspace=alt, ...)
```

See [`examples/showcase.py`](examples/showcase.py) for workspace persistence,
session reads, time travel through `list[Graph]`, and gym-style stepping in one
file.

## Rich visualization

See [notebook](./examples/notebooks/viz_walkthrough.ipynb) for a full showcase of vizualization utilities.

Because the run is a typed graph, every visualization is just a render of
that graph. Normal runs are viewed from their workspace.


### Gradio viewer

![](docs/static/gradio_ui.png)

`open_viewer(workspace)` launches a small browser app for inspecting a
saved workspace — tree, summary, and raw state JSON side by side:

```python
from rlmflow.utils.viewer import open_viewer

open_viewer("./myproject")
```

From the CLI: `rlmflow view ./myproject --port 7861`.

### Live terminal tree

`rlmflow.utils.viz.live(agent, graph)` drives the step loop and renders a
Rich tree as states are produced. The boids run (`Create a simple boids
simulation in plain HTML and JavaScript, split each component into
separate files`) settles to:

```text
root [result] {default:gpt-5} -> Boids simulation written to output/boids-simulation with modular JS (boid, simulation, renderer) and index.html entrypoint.
  root.index_html    [result] {fast:gpt-5-mini} -> ok
  root.styles_css    [result] {fast:gpt-5-mini} -> ok
  root.boid_js       [result] {fast:gpt-5-mini} -> ok
  root.simulation_js [result] {fast:gpt-5-mini} -> ok
  root.renderer_js   [result] {fast:gpt-5-mini} -> ok
  root.main_js       [result] {fast:gpt-5-mini} -> ok
```

The same render is available offline as `graph.tree()` on any snapshot.
Filename-flavored agent ids (`index.html` → `index_html`) are sanitized
because `.` is the parent/child delimiter in the agent tree.

### Static renders

`rlmflow render <path> -f F` writes a static visualization in any of:

```text
mermaid             # stateDiagram-v2 (default topology)
mermaid-flowchart   # flowchart TD, better for wide trees
mermaid-sequence    # sequenceDiagram of delegate / wait / resume
dot · d2            # Graphviz / D2 topology
tree · ascii-boxes  # text trees
gantt-html          # standalone HTML swimlane
report-md           # full Markdown summary (tree + cost + result + errors)
code-log            # every code block paired with its observation
error-summary       # ErrorOutput counts grouped by kind
tokens              # one-line ASCII sparkline of cumulative tokens
html                # self-contained interactive stepper, one slide per snapshot
image               # single PNG/SVG/PDF of the topology snapshot
steps               # one image per snapshot, written as step_NN.{png,svg,pdf}
```

```bash
rlmflow render ./myproject -f mermaid-flowchart
rlmflow render ./myproject -f gantt-html -o run.html
rlmflow render ./myproject -f report-md  -o run.md
rlmflow render ./myproject -f tokens
```

GitHub renders mermaid inline, so the output drops straight into a doc.
The example below is the `to_mermaid_flowchart(graph)` projection of the
boids run; it renders reliably across the GitHub-supported mermaid
versions:

```mermaid
flowchart TD
    root["root<br/><i>result</i><br/>Boids simulation written to output/boids-simulation..."]:::result
    root --> html["root.index_html<br/><i>result</i><br/>ok"]:::result
    root --> css["root.styles_css<br/><i>result</i><br/>ok"]:::result
    root --> boid["root.boid_js<br/><i>result</i><br/>ok"]:::result
    root --> sim["root.simulation_js<br/><i>result</i><br/>ok"]:::result
    root --> rend["root.renderer_js<br/><i>result</i><br/>ok"]:::result
    root --> main["root.main_js<br/><i>result</i><br/>ok"]:::result
    classDef result fill:#3fb95022,stroke:#3fb950,color:#c9d1d9;
```

### Programmatic helpers

Everything the CLI does is one function call away:

```python
from rlmflow.utils.export import to_mermaid, to_mermaid_flowchart, to_mermaid_sequence, to_dot, to_d2
from rlmflow.utils.viz import (
    ascii_boxes, code_log, error_summary, message_stream, diff_system_prompts,
    gantt, gantt_html, token_sparkline, budget_burndown, bench_table,
    report_md, live, tee, slack_webhook, discord_webhook,
)
from rlmflow.utils.tracing import json_logs

print(token_sparkline(graphs))          # ▁▂▅█▂   15820 tok over 7 steps
print(error_summary(graph))             # ErrorOutput counts grouped by kind
print(message_stream("root.boid_js", graph))     # rendered transcript for one agent
print(report_md(graphs, title="run"))   # full Markdown report
gantt_html(graphs, "run.html")          # standalone HTML swimlane
json_logs(graph, "run.jsonl")           # one state per line
```

### Image, GIF, and HTML exports

For blog posts, PR comments, papers, and CI artifacts, render the
graph straight to a PNG/SVG/PDF, an animated GIF, or a single
self-contained HTML stepper. Four public functions live in
`rlmflow.utils`, plus matching CLI verbs:

| Function                                | CLI verb        | Output                                | Use case                                   |
|-----------------------------------------|-----------------|---------------------------------------|--------------------------------------------|
| `save_image(graph, path)`               | `-f image`      | one PNG/SVG/PDF                       | hero image of a finished run               |
| `save_steps(graphs, dir/)`              | `-f steps`      | `step_NN.png` per snapshot            | blog slideshow, paper figure series        |
| `save_gif(graphs, path)`                | _(no verb yet)_ | animated GIF                          | quick preview / social posts               |
| `save_html(graphs, path)`               | `-f html`       | self-contained stepper (Plotly + CSS) | shareable URL-less artifact, PR comment    |

Quick start:

```python
from rlmflow import Workspace
from rlmflow.utils import save_image, save_steps, save_html, save_gif

workspace = "./myproject"
graph = Workspace.open_path(workspace).load_graph()

save_image(workspace, "run_final.png")           # latest workspace snapshot
save_html(workspace, "viewer.html", title="run") # standalone viewer

# If you kept an in-memory history list, playback exports still work:
save_steps(graphs, "frames/")                    # one PNG per step
save_gif(graphs, "trace.gif", duration=400)      # animated GIF (~2.5 fps)
```

Or use the graph shorthand (same defaults):

```python
graph.save_image("run_final.png")
graph.save_html("viewer.html")
```

#### Why the scaling knobs exist

The Plotly viewer, static image export, GIF export, and HTML stepper now
share the same default element scale (`element_mult=1.0`), so a saved
PNG looks much closer to the Gradio/Jupyter view. Dense graphs still
adaptively cap marker and label sizes to avoid turning large runs into
solid blobs.

Use these knobs only when a target medium needs a different balance:

| Knob               | Default | Effect                                                                                  |
|--------------------|---------|------------------------------------------------------------------------------------------|
| `element_mult`     | `1.0`   | Uniform multiplier on markers and fonts. The simplest "make it bigger" knob.            |
| `marker_mult`      | _(inherits)_ | Override just marker size and outline width. Useful when dots need more visual weight. |
| `text_mult`        | _(inherits)_ | Override just label font size. Smaller text means fewer label collisions.              |
| `normalize_labels` | `True`  | Force every label to `bottom center` so adjacent depths can't share a vertical band.     |

Pass `marker_mult` and/or `text_mult` to break the symmetry when labels
are colliding or nodes are too subtle for a specific export.

#### Recipes

**Hero PNG of a finished run** — defaults are tuned for this:

```python
graph.save_image("hero.png")
# == save_image(graph, "hero.png", width=1800, height=1350,
#               scale=2.0, element_mult=1.0, normalize_labels=True)
```

**Blog slideshow with dense subtrees** — fat markers, small labels,
square-ish canvas (the recipe behind `docs/blog.md`):

```python
save_steps(
    graphs,
    "blog/frames/",
    width=1600, height=1200, scale=2.0,
    marker_mult=3.5,        # fat node dots + edges
    text_mult=2.2,          # shrink labels so they don't collide
    normalize_labels=True,  # already the default — explicit for the reader
)
```

**Standalone interactive stepper** — drop into a PR comment or
GitHub gist:

```python
save_html(workspace, "viewer.html", title="needle haystack run")
```

The HTML output embeds Plotly from CDN, includes per-slide
transcripts, and ships keyboard navigation (← / →) plus dot-style
slide indicators. Open it in any browser, attach it to an email,
upload it as a CI artifact — it works offline once the CDN script
is cached.

**Animated GIF** — needs `pip install rlmflow[image] pillow`:

```python
save_gif(
    graphs,
    "trace.gif",
    duration=600,          # ms per frame; lower = faster
    loop=0,                # 0 = forever; 1 = play once
    width=1200, height=900,
)
```

#### From the CLI

Every knob above maps 1:1 to a CLI flag:

```bash
# blog slideshow recipe (matches the dense-tree recipe above)
rlmflow render ./myproject \
  -f steps -o blog/frames/ \
  --width 1600 --height 1200 --scale 2.0 \
  --marker-mult 3.5 --text-mult 2.2

# self-contained interactive stepper
rlmflow render ./myproject \
  -f html  -o stepper.html --title "boids walkthrough"

# single hero PNG with default scaling
rlmflow render ./myproject \
  -f image -o hero.png

# opt out of label normalization (matches Gradio viewer defaults)
rlmflow render ./myproject \
  -f html  -o stepper.html --no-normalize-labels
```

The CLI uses `element_mult=1.0` by default for `html`, `image`, `steps`,
and `gif` so static exports stay visually consistent with the interactive
viewer. Node sizes are uniform; token counts stay in hover/details, not
marker size. Override with `--element-mult`, `--marker-mult`, or
`--text-mult` for a specific medium.

#### Dependencies

- `save_image` / `save_steps` need `kaleido`. Install with
  `pip install rlmflow[image]` or just `pip install kaleido`.
- `save_gif` additionally needs `Pillow`
  (`pip install rlmflow[image] pillow`).
- `save_html` and `render_html` have **no static-image dependency** —
  they emit a single HTML file that embeds Plotly from CDN.

## DSPy Adapter

`RLMFlowLM` lets DSPy use an `RLMFlow` agent anywhere it expects a
language model:

```python
from pathlib import Path

import dspy

from rlmflow import OpenAIClient, RLMConfig, RLMFlow, Workspace
from rlmflow.integrations.dspy import RLMFlowLM
from rlmflow.runtime.local import LocalRuntime

workspace = Workspace.create(Path("examples/example-workspaces/dspy-workspace"))
agent = RLMFlow(
    llm_client=OpenAIClient("gpt-4o-mini"),
    runtime=LocalRuntime(workspace=workspace),
    config=RLMConfig(max_depth=1, max_iterations=5),
)

dspy.configure(lm=RLMFlowLM(agent, model="rlmflow/gpt-4o-mini"))
qa = dspy.ChainOfThought("question -> answer")
print(qa(question="What is 17 * 23?").answer)
```

Install it with `pip install rlmflow[openai,dspy]`. See
[`examples/dspy_drop_in.py`](examples/dspy_drop_in.py) for the runnable
version.

## Examples

All examples share flags like `--no-viz`, `--docker-image rlmflow:local`,
`--max-depth`, and `--max-iterations`. See [`examples/README.md`](examples/README.md).

| Example | What it shows |
|---|---|
| [`showcase.py`](examples/showcase.py) | `Graph` snapshots, workspace persistence, session reads, time travel, gym-style stepping. |
| [`drop_in_llm.py`](examples/drop_in_llm.py) | `RLMFlow` as an `LLMClient`. Nested agents. |
| [`dspy_drop_in.py`](examples/dspy_drop_in.py) | Use an `RLMFlow` agent as the LM behind a DSPy program. |
| [`sandbox/`](examples/sandbox/) | Build a small web app whose Python code runs inside Modal, E2B, and Daytona sandboxes. |
| [`coding-agent/agent.py`](examples/coding-agent/agent.py) | Interactive coding agent that writes and edits files. |
| [`needle_haystack.py`](examples/needle_haystack.py) | Needle-in-a-haystack over a massive in-memory `CONTEXT`, using parallel child chunks. |
| [`needle_haystack_filesystem.py`](examples/needle_haystack_filesystem.py) | Needle-in-a-haystack across many files with custom tools and `runtime_factory`. |
| [`summarizer.py`](examples/summarizer.py) | Recursive map-reduce summarization over a long document — `launch_subagents` fan-out + stateful combine. |
| [`eager_children.py`](examples/eager_children.py) | `eager_children=True` vs `False` — how child scheduling overlaps. |
| [`fork_repair.py`](examples/fork_repair.py) | Fork a workspace into independent repair branches and run tests in each. |
| [`best_of_n.py`](examples/best_of_n.py) | Run N independent workspace branches and pick the best result. |
| [`autoresearch/`](examples/autoresearch/) | Karpathy-style hill-climbing research loop with custom `@tool`s and delegation. |
| [`graph-features/`](examples/graph-features/) | Offline tour of the `Graph` API: query, navigate, mutate, save/load, replay, fork, render. |
| [`view_demo.py`](examples/view_demo.py) | Build synthetic `Graph` snapshots and launch the Gradio viewer. |
| [`notebooks/coding_agent.ipynb`](examples/notebooks/coding_agent.ipynb) | Build the agent, run the boids task end-to-end, and inspect the workspace/viewer. Requires a live LLM. |
| [`notebooks/viz_walkthrough.ipynb`](examples/notebooks/viz_walkthrough.ipynb) | Every visualization against the saved fixture: inline tree, Plotly graph, HTML stepper, topology renders (mermaid/dot/d2/sequence), step-indexed timeline, per-state detail, cost & reports, run-vs-run comparison, CLI equivalents. |
| [`notebooks/node_basics.ipynb`](examples/notebooks/node_basics.ipynb) | `Graph` query API tour — `graph[aid]`, `graph.nodes`, `graph.nodes.find`, `graph.nodes.where`, `graph.nodes.results`/`errors`, per-agent tokens, `session.load_graph`, state streaming with `tee` / `json_logs`. |

## Benchmarks

A runnable RLM-vs-flat harness for **OOLONG** (long-context aggregation,
~250k tokens) lives under [`benchmarks/oolong/`](benchmarks/oolong/).
It mirrors Prime Intellect's reference environment but talks directly
to `rlmflow` instead of `verifiers`. Three modes — `standard` (one big
flat call), `rlm` (recursive scaffold), `rlm_tips` (recursive +
chunking hints) — across `synth`, `synth_with_labels`, and `real`
subsets, scored deterministically against the published gold answers.

```bash
python benchmarks/oolong/run.py --mode rlm --subset synth --limit 50
python benchmarks/oolong/aggregate.py --runs runs/oolong-*
```

See [`benchmarks/oolong/README.md`](benchmarks/oolong/README.md) for
flags, scoring details, and ablation scripts.

## CLI

```
rlmflow view ./myproject
rlmflow render ./myproject -f mermaid
rlmflow render ./myproject -f gantt-html -o run1.html
rlmflow render ./myproject -f html       -o stepper.html
rlmflow render ./myproject -f steps      -o frames/  --marker-mult 3.5 --text-mult 2.2
rlmflow render ./myproject -f image      -o graph.png
rlmflow version
```

`view` and `render` accept a workspace directory.
`render -f` accepts: `mermaid`, `mermaid-flowchart`, `mermaid-sequence`,
`dot`, `d2`, `tree`, `ascii-boxes`, `gantt-html`, `report-md`, `code-log`,
`error-summary`, `tokens`, `html`, `image`, `steps` — see the
[Static renders](#static-renders) table and [Image, GIF, and HTML
exports](#image-gif-and-html-exports) for what each produces and the
scaling / label-normalization flags (`--marker-mult`, `--text-mult`,
`--normalize-labels` / `--no-normalize-labels`).

## Roadmap
- [x] OOLONG long-context aggregation harness (`standard` / `rlm` / `rlm_tips`)
- [x] `LocalRuntime` + `DockerRuntime` — battle-tested
- [~] `ModalRuntime` / `E2BRuntime` / `DaytonaRuntime` — full support: native SDK file transfer, real-sandbox CI, depth>1 delegation, heavier example
- [~] Depth × breadth sweep (accuracy vs `max_depth`) on an OOLONG/BABILong subset
- [~] [RAO](https://apga.github.io/RAO/) (recursive agent optimization) support
- [ ] OOLONG-Pairs harness — pairwise aggregation where flat LLMs fall over
- [ ] Fork/resume determinism test in CI to guard the replay machinery
- [ ] LongBench-v2 CodeQA + SWE-bench adapter for coding-agent credibility

## Docs

The top-level docs are short, user-facing guides. The deep dive lives
in [`docs/internals.md`](docs/internals.md).

- [**Internals**](docs/internals.md): deep reference — engine
  architecture, step lifecycle (`act` → `apply_one`), the REPL `await`
  protocol, resume semantics, cold-start replay, persistence, and the
  full `RLMFlow` override surface. Start here if you want to subclass
  the engine.
- [Blog post](docs/blog.md): long-form pitch — why recursive language
  models, why graphs over flat traces, full needle-in-a-haystack
  walkthrough with the same exports the CLI ships.
- [Positioning](docs/positioning.md): when to use rlmflow vs
  rlm-minimal, ypi, LangGraph, CrewAI, AutoGen, SWE-agent, Aider.
- [Control](docs/control.md): step loop, workspace resume, rewind,
  forks, `CONTEXT.read()` / slices, `launch_subagent` / `launch_subagents`,
  inline-first strategy, custom tools.
- [Node injection](docs/injections.md): append typed controller events to a
  running graph and commit them through `agent.step(graph)`.
- [Observability](docs/observability.md): querying the `Graph`,
  workspace layout, export helpers, live tree, gantt, topology
  exports, Gradio viewer, CLI.
- [Runtimes](docs/runtimes.md): `Runtime` protocol, shipped runtimes
  (Local / Docker / Modal / E2B / Daytona), writing your own.
- [Prompt customization](docs/prompt_customization.md): `PromptBuilder`
  sections, deriving from the default prompt, full replacement.
- [Security](docs/security.md): trust model, Docker isolation knobs,
  engine-level caps, proxied tools, approval gates.
- [Changelog](CHANGELOG.md): release-by-release changes.

## References

- [Recursive Language Models](https://github.com/alexzhang13/rlm): the
  original RLM paper and implementation.
- [rlm-minimal](https://github.com/alexzhang13/rlm-minimal): the
  single-file reference rlmflow grew from.
- [Scaling Managed Agents: Decoupling the brain from the hands](https://www.anthropic.com/engineering/managed-agents):
  Anthropic's writeup on separating harness, session, and sandbox
  interfaces for long-horizon agents.
- [ypi](https://github.com/rawwerks/ypi): recursive coding agent built
  on Pi. Our session layout and much of the default prompt
  (size-up → delegate → combine, guardrails, aggressive delegation) come
  from ypi's `SYSTEM_PROMPT.md`.

## License

See [LICENSE](LICENSE).

## Citation

```bibtex
@misc{sudhakaran2025rlmflow,
  author = {Sudhakaran, Shyam},
  title = {rlmflow},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/shyamsn97/rlmflow}},
}
```
