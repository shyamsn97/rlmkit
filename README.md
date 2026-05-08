# rlmflow

<p align="center">
  <a href="https://pypi.org/project/rlmflow/"><img src="https://img.shields.io/pypi/v/rlmflow.svg?label=pypi" alt="PyPI" /></a>
  <a href="https://github.com/shyamsn97/rlmflow/pkgs/container/rlmflow"><img src="https://img.shields.io/badge/ghcr.io-rlmflow-2496ED?logo=docker&logoColor=white" alt="Docker" /></a>
</p>

A Python library for createing interactible, steppable graph [Recursive Language Models](https://arxiv.org/abs/2512.24601).

Recursive Language Models are powerful systems -- capable of handling long-context tasks by spawning sub-agents with their own fresh context windows. However. RLMs get messy fast: parents spawn children, children spawn more children, which also can run for multiple steps, etc.

**rlmflow** turns the run into an explicit graph. Every query, action,
observation, child call, wait, resume, and result is a typed, immutable
node you can step, inspect, fork, and replay.

<p align="center">
  <img src="docs/rlm_animation.gif" alt="rlmflow animation" />
</p>

## RLMs as Graphs

RLMs delegate subtasks to
children, those children can delegate to their own children, and their results bubble back up. **rlmflow** represents this representation as a tree directory: every step inside an agent is a typed node and every
delegation is an edge between agents.

For example, this RLM code:

```python
h1 = delegate("search", "Find evidence", context=chunk_a)
h2 = delegate("verify", "Check the answer", context=chunk_b)
results = yield wait(h1, h2)
done(combine(results))
```

becomes this execution graph:

```text
Query(root)
  -> Action(root: delegate search + verify)
  -> Supervising(root: waiting on search, verify)
      -> Query(root.search)
      -> Result(root.search)
      -> Query(root.verify)
      -> Result(root.verify)
  -> Resume(root: search + verify results)
  -> Result(root)
```

## Install

```
pip install rlmflow               # core
pip install rlmflow[openai]       # + OpenAI client
pip install rlmflow[anthropic]    # + Anthropic client
pip install rlmflow[viewer]       # + Gradio viewer
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
from rlmflow.utils.trace import save_trace
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
states = [agent.start(query)]
while not states[-1].finished:
    states.append(agent.step(states[-1]))
    print(states[-1].tree())

save_trace(states, "traces/run1")
open_viewer(states)
```

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

`step(node) -> node'` is one atomic graph transition. Every step returns a
new immutable `Node`, so the live tree is just `state.tree()`:

```python
state = agent.start(query)
while not state.finished:
    state = agent.step(state)
print(state.tree())
```

```text
root [supervising] {default}
├── root.scanner_auth [result] {fast} -> Found SQL injection in login.py
├── root.scanner_api  [supervising] {default}
│   ├── root.scanner_api.chunk_0 [result] {fast} -> Clean
│   └── root.scanner_api.chunk_1 [result] {fast} -> Payment flow is safe
└── root.scanner_db   [result] {fast} -> No issues found
```

Every transition follows the same shape:

```text
Observation -> LLM -> Action -> Runtime -> Observation     (REPL output)
                              -> done()  -> Result          (terminal answer)
                              -> wait()  -> Supervising     (waiting on children)
Supervising -> children done -> Resume   -> LLM  -> ...
```

`Observation`, `Action`, `Supervising`, `Resume`, and `Result` are all
typed Pydantic nodes. The graph is queryable in plain Python:

```python
state.tree()                                  # ASCII render
state.find("root.scanner_api")                # one node by id or agent_id
state.path_to("root.scanner_api.chunk_1")     # root → node ancestor chain
state.leaves()                                # every node with no children
state.errors()                                # every ErrorNode in the subtree
state.results()                               # every ResultNode in the subtree
state.where(type="action", agent_id="root")   # kwargs match node attrs
state.where(lambda n: n.depth > 2)            # or pass a predicate
state.model_dump_json()                       # full serialization
```

## Checkpoint, branch, replay

Every node is a frozen Pydantic snapshot, so the whole run is data:

```python
from rlmflow import Node

state.save(workspace.checkpoint_path)

# resume later, in another process, with a different model
state = Node.load(workspace.checkpoint_path)
agent = RLMFlow(llm_client=AnotherModel(), workspace=workspace, ...)
while not state.finished:
    state = agent.step(state)
```

To branch into an isolated workspace with its own session, context, and
working tree:

```python
alt = workspace.fork(new_branch_id="repair", new_dir="./runs/repair")
alt_agent = RLMFlow(llm_client=..., workspace=alt, ...)
```

Or intervene mid-run by replacing a child node before the parent resumes —
see [`examples/showcase.py`](examples/showcase.py) for checkpointing,
time travel, manual intervention, and gym-style stepping in one file.

## Rich visualization

See [notebook](./examples/notebooks/viz_walkthrough.ipynb) for a full showcase of vizualization utilities.

Because the run is a typed graph, every visualization is just a render of
that graph. The coding agent example
([`examples/coding-agent/agent.py`](examples/coding-agent/agent.py))
already exercises every option below — its saved trace under
`examples/data/notebook-coding-agent/` is the source for the renders here.


### Gradio viewer

![](docs/static/gradio_ui.png)

`open_viewer(states)` launches a small browser app for stepping through a
saved trace — tree, summary, and raw node JSON side by side:

```python
from rlmflow.utils.trace import load_trace
from rlmflow.utils.viewer import open_viewer

trace = load_trace("examples/data/notebook-coding-agent/trace")
open_viewer(trace.states)
```

Or from a checkpoint via the CLI: `rlmflow view examples/data/notebook-coding-agent/trace`.

### Live terminal tree

`rlmflow.utils.viz.live(agent, state)` drives the step loop and renders a
Rich tree as nodes are produced. The boids run (`Create a simple boids
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

The same render is available offline as `state.tree()` on any node.
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
error-summary       # ErrorNode counts grouped by kind
tokens              # one-line ASCII sparkline of cumulative tokens
```

```bash
rlmflow render examples/data/notebook-coding-agent/trace -f mermaid-flowchart
rlmflow render examples/data/notebook-coding-agent/trace -f gantt-html -o run.html
rlmflow render examples/data/notebook-coding-agent/trace -f report-md  -o run.md
rlmflow render examples/data/notebook-coding-agent/trace -f tokens
```

GitHub renders mermaid inline, so the output drops straight into a doc.
The example below is the `to_mermaid_flowchart(state)` projection of the
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

print(token_sparkline(states))          # ▁▂▅█▂   15820 tok over 7 steps
print(error_summary(state))             # ErrorNode counts grouped by kind
print(message_stream("root.boid_js", session))   # rendered transcript for one agent
print(report_md(states, title="run"))   # full Markdown report
gantt_html(states, "run.html")          # standalone HTML swimlane
json_logs(states, "run.jsonl")          # one node per line
```

## Examples

All examples share flags like `--no-viz`, `--docker-image rlmflow:local`,
`--max-depth`, and `--max-iterations`. See [`examples/README.md`](examples/README.md).

| Example | What it shows |
|---|---|
| [`showcase.py`](examples/showcase.py) | Typed nodes, checkpoints, session persistence, intervention, gym-style stepping. |
| [`drop_in_llm.py`](examples/drop_in_llm.py) | `RLMFlow` as an `LLMClient`. Nested agents. |
| [`coding-agent/agent.py`](examples/coding-agent/agent.py) | Interactive coding agent that writes and edits files. |
| [`needle_haystack.py`](examples/needle_haystack.py) | Needle-in-a-haystack across 500 files with custom tools and `runtime_factory`. |
| [`summarizer.py`](examples/summarizer.py) | Recursive map-reduce over a long document. |
| [`view_demo.py`](examples/view_demo.py) | Launch the Gradio viewer on a saved trace. |
| [`notebooks/coding_agent.ipynb`](examples/notebooks/coding_agent.ipynb) | Build the agent, run the boids task end-to-end, open the interactive viewer. **Source of `examples/data/notebook-coding-agent/`** — every other notebook reads from here. |
| [`notebooks/viz_walkthrough.ipynb`](examples/notebooks/viz_walkthrough.ipynb) | All 9 visualizations against the saved boids trace: inline tree, interactive viewer, topology renders (mermaid/dot/d2/sequence), step-indexed timeline, per-node detail (`message_stream`, `diff_system_prompts`), cost & reports, run-vs-run comparison, CLI equivalents. |
| [`notebooks/node_basics.ipynb`](examples/notebooks/node_basics.ipynb) | `Node` API tour — walk, find, path_to, filter (`leaves`/`results`/`errors`/`where`), diff snapshots, session access (`FileSession.load`, `chain_to`), event streaming with `tee` / `json_logs`. |

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
rlmflow view traces/run1/
rlmflow render checkpoint.json -f mermaid
rlmflow render traces/run1/ -f gantt-html -o run1.html
rlmflow version
```

`view` and `render` accept a trace directory, `trace.json`, or checkpoint.
`render -f` accepts: `mermaid`, `mermaid-flowchart`, `mermaid-sequence`,
`dot`, `d2`, `tree`, `ascii-boxes`, `gantt-html`, `report-md`, `code-log`,
`error-summary`, `tokens` — see the [Static renders](#static-renders)
table above for what each produces.

## Todo


## Docs
- [Positioning](docs/positioning.md): when to use rlmflow vs rlm-minimal,
  ypi, LangGraph, CrewAI, AutoGen, SWE-agent, Aider — decision matrix and
  per-framework comparisons.
- [Observability](docs/observability.md): node fields and types, save/load
  traces, session/context layout, live tree, gantt, topology exports,
  Gradio viewer, CLI.
- [Control](docs/control.md): step loop, checkpoint, rewind, fork
  (`Workspace.fork`, `CONTEXT.fork()`), `delegate(name, query, context)`,
  inline-first strategy, intervention, custom prompts, runtimes, tools.
- [Runtimes](docs/runtimes.md): `Runtime` protocol, shipped runtimes
  (Local / Subprocess / Docker / Modal), writing your own.
- [Security](docs/security.md): trust model, Docker isolation knobs,
  engine-level caps, proxied tools, approval gates.
- [Changelog](CHANGELOG.md): release-by-release changes, including the
  upcoming `delegate(...)` mandatory-`context` break.

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
