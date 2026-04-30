# rlmflow

<p align="center">
  <a href="https://pypi.org/project/rlmflow/"><img src="https://img.shields.io/pypi/v/rlmflow.svg?label=pypi" alt="PyPI" /></a>
  <a href="https://github.com/shyamsn97/rlmflow/pkgs/container/rlmflow"><img src="https://img.shields.io/badge/ghcr.io-rlmflow-2496ED?logo=docker&logoColor=white" alt="Docker" /></a>
</p>

A Python library for building [Recursive Language Models](https://github.com/alexzhang13/rlm-minimal)
as inspectable execution graphs.

Recursive agents get messy fast: one parent spawns children, children spawn
more children, some branches wait, some fail, and some resume later with
partial results. A flat chat log hides that structure.

**rlmflow** turns the run into a graph instead, where every query, action, observation, child call, etc. is a typed node you can inspect, visualize, step, fork, and replay.

<p align="center">
  <img src="docs/rlm_animation.gif" alt="rlmflow animation" />
</p>

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

## RLMFlow as a Graph Engine

`RLMFlow` implements `LLMClient`, so it is a drop-in replacement for any LLM.
Call `chat(messages)` or `run(query)` and it runs a recursive agent loop
underneath:

```python
def ask(llm: LLMClient, q: str) -> str:
    return llm.chat([{"role": "user", "content": q}])

ask(OpenAIClient("gpt-4o-mini"), "2+2?")             # one LLM call
ask(RLMFlow(llm_client=..., runtime=...), "2+2?")    # full agent, same return type
```

Nest agents by passing one `RLMFlow` as another's `llm_client`.

The runtime model stays small:

- `RLMFlow` is the lightweight interpreter that advances nodes.
- `Node` objects are the source of truth for what happened.
- `Workspace.session` stores node/message history under `session/`.
- `Workspace.context` stores task payloads exposed in the REPL as `CONTEXT`.
- `Runtime` owns live REPL execution state.

Each transition returns a new immutable node. A live tree might look like:

```
root [supervising] {default}
├── root.scanner_auth [result] {fast:gpt-5-mini} -> Found SQL injection in login.py
├── root.scanner_api [supervising] {default}
│   ├── root.scanner_api.chunk_0 [result] {fast:gpt-5-mini} -> Clean
│   └── root.scanner_api.chunk_1 [result] {fast:gpt-5-mini} -> Payment flow is safe
└── root.scanner_db [result] {fast:gpt-5-mini} -> No issues found
```

Each `step(node) -> node'` is one atomic graph transition:

```
ObservationNode -> LLM -> ActionNode -> Runtime -> ObservationNode
                                      -> done() -> ResultNode
                                      -> wait() -> SupervisingNode
SupervisingNode -> step child leaves -> ResumeNode -> LLM -> ...
```

- `ObservationNode`: input to the next LLM call: query, REPL output, resume,
  error, or terminal result.
- `ActionNode`: raw LLM reply plus extracted REPL code.
- `SupervisingNode`: suspended action waiting on child agents.
- `ResultNode`: terminal answer from `done(result)`.

Delegation:

```python
h1 = delegate("searcher", "Find all TODOs in src/")
h2 = delegate("searcher", "Find all FIXMEs in src/")   # auto-suffixed
results = yield wait(h1, h2)
done(f"Found {len(results)} batches")
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

- [RLMs Are Graphs](docs/internal/rlms_are_graphs.md): the design thesis:
  typed execution graphs, flow interpreter, session/context/runtime split.
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
