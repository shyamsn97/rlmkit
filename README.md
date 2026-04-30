# rlmflow

<p align="center">
  <a href="https://pypi.org/project/rlmflow/"><img src="https://img.shields.io/pypi/v/rlmflow.svg?label=pypi" alt="PyPI" /></a>
  <a href="https://github.com/shyamsn97/rlmflow/pkgs/container/rlmflow"><img src="https://img.shields.io/badge/ghcr.io-rlmflow-2496ED?logo=docker&logoColor=white" alt="Docker" /></a>
</p>

A Python library for [Recursive Language Models](https://arxiv.org/abs/2512.24601) —
a typed graph for every recursive run.

Recursive Language Models are powerful systems -- capable of handling long-context tasks by spawning sub-agents with their own fresh context windows. However. RLMs get messy fast: parents spawn children, children spawn more children, which also can run for multiple steps, etc.

**rlmflow** turns the run into an explicit graph. Every query, action,
observation, child call, wait, resume, and result is a typed, immutable
node you can step, inspect, fork, and replay.

<p align="center">
  <img src="docs/rlm_animation.gif" alt="rlmflow animation" />
</p>

## RLMs are Graphs

An RLM run is a tree of agents. A parent agent delegates subtasks to
children, those children can delegate to their own children, and waits
and resumes connect their results back up. rlmflow represents that tree
directly: every step inside an agent is a typed node, and every
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

This example is all you need for a simple and interpretable recursive coding agent.

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
    config=RLMConfig(max_depth=3, max_iterations=15),
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
typed Pydantic nodes. `state.tree()`, `state.find(agent_id)`, and
`state.model_dump_json()` are just attribute access on the graph.

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

## CLI

```
rlmflow view traces/run1/
rlmflow render checkpoint.json -f mermaid
rlmflow render traces/run1/ -f gantt-html -o run1.html
rlmflow version
```

`view` and `render` accept a trace directory, `trace.json`, or checkpoint.
Formats: `mermaid`, `dot`, `tree`, `gantt-html`.

## Docs
- [Positioning](docs/positioning.md): why rlmflow treats RLMs as graphs.
- [Observability](docs/observability.md): nodes, session/context, traces,
  visualizations, and the viewer.
- [Control](docs/control.md): step loop, checkpoint, rewind, intervention,
  custom prompts, runtimes, and tools.
- [Runtimes](docs/runtimes.md): `Runtime` protocol, Local / Subprocess
  / Docker / Modal, writing your own.
- [Security](docs/security.md): trust model, Docker isolation knobs,
  approval gates.

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
