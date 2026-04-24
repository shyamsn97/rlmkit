# rlmkit

A minimal state-machine library for [Recursive Language Model](https://github.com/alexzhang13/rlm-minimal)
agents. Every agent — root and descendants — advances one step at a
time, and the entire computation tree is a single immutable,
serializable object at every step boundary.

<p align="center">
  <img src="docs/rlm_animation.gif" alt="rlmkit animation" />
</p>

## Install

```
pip install rlmkit               # core
pip install rlmkit[openai]       # + OpenAI client
pip install rlmkit[anthropic]    # + Anthropic client
pip install rlmkit[viewer]       # + Gradio viewer
pip install rlmkit[all]          # all of the above
```

From source:

```
git clone https://github.com/shyamsn97/rlmkit && cd rlmkit
pip install -e .
```

## Quick start

```python
from rlmkit import RLM, RLMConfig, OpenAIClient
from rlmkit.runtime.local import LocalRuntime
from rlmkit.tools import FILE_TOOLS
from rlmkit.utils.viewer import save_trace, open_viewer

runtime = LocalRuntime(workspace="./myproject")
runtime.register_tools(FILE_TOOLS)

agent = RLM(
    llm_client=OpenAIClient("gpt-5"),
    runtime=runtime,
    config=RLMConfig(max_depth=3, max_iterations=15, session="context"),
)

query = "Build a python text-based adventure game with combat and inventory."
states = [agent.start(query)]
while not states[-1].finished:
    states.append(agent.step(states[-1]))
    print(states[-1].tree())

save_trace(states, "traces/run1", query=query)
open_viewer(states, query=query)
```

## Examples

All examples share the same CLI flags — `--no-viz`, `--docker-image rlmkit:local`,
`--max-depth`, etc. See [`examples/README.md`](examples/README.md).

| Example | What it shows |
|---|---|
| [`showcase.py`](examples/showcase.py) | Checkpoint, fork, session persistence, time travel, intervention — the full API tour. |
| [`drop_in_llm.py`](examples/drop_in_llm.py) | `RLM` as an `LLMClient`. Nested agents. |
| [`coding-agent/agent.py`](examples/coding-agent/agent.py) | Interactive coding agent that writes and edits files. |
| [`needle_haystack.py`](examples/needle_haystack.py) | Needle-in-a-haystack across 500 files with custom tools and `runtime_factory`. |
| [`summarizer.py`](examples/summarizer.py) | Recursive map-reduce over a 10k-line document. |
| [`view_demo.py`](examples/view_demo.py) | Launch the Gradio viewer on a saved trace. |

## Overview

`RLM` implements `LLMClient` — it's a drop-in replacement for any LLM.
Call `chat(messages)` or `run(query)` and it runs a full recursive
agent loop underneath. Swap your LLM for an RLM and get delegation,
parallel sub-agents, and a code REPL for free.

```python
def ask(llm: LLMClient, q: str) -> str:
    return llm.chat([{"role": "user", "content": q}])

ask(OpenAIClient("gpt-4o-mini"), "2+2?")             # one LLM call
ask(RLM(llm_client=..., runtime=...), "2+2?")        # full agent, same return type
```

Nest agents by passing one `RLM` as another's `llm_client`.

The tree is a **state machine**. Every agent advances one step at a
time, and the full computation is one object you can inspect,
checkpoint, fork, or serialize at any step boundary:

```
root [supervising] iter 5
├── root.scanner_auth [finished] iter 3 → "Found SQL injection in login.py"
│   ├── root.scanner_auth.chunk_0 [finished] iter 2 → "No issues"
│   ├── root.scanner_auth.chunk_1 [finished] iter 2 → "SQL injection on line 42"
│   └── root.scanner_auth.chunk_2 [finished] iter 2 → "No issues"
├── root.scanner_api [supervising] iter 3
│   ├── root.scanner_api.chunk_0 [ready] iter 1
│   ├── root.scanner_api.chunk_1 [finished] iter 2 → "Clean"
│   │   └── root.scanner_api.chunk_1.deep_scan [finished] iter 2 → "Payment flow is safe"
│   └── root.scanner_api.chunk_2 [finished] iter 2 → "Clean"
└── root.scanner_db [finished] iter 2 → "No issues found"
```

Each `step(state) -> state'` is one atomic transition:

```
        step_llm()              step_exec()
READY ─────────────> EXECUTING ─────────────> SUPERVISING
  ^                      |                        |
  |                   done()                step_children()
  |                      |                   (one batch)
  |                      v                        |
  |                  FINISHED <── resume_exec() ──┤
  |                      ^                        |
  +── yields again ──────┘                  children not done
```

- **READY** — queued for the next LLM call.
- **EXECUTING** — LLM returned a ` ```repl ``` ` block; the engine runs it.
- **SUPERVISING** — code called `delegate()` + `yield wait()`; children
  are running. Each `step()` advances children by one batch.
- **FINISHED** — code called `done(result)`.

Delegation:

```python
h1 = delegate("searcher", "Find all TODOs in src/")
h2 = delegate("searcher", "Find all FIXMEs in src/")   # auto-suffixed
results = yield wait(h1, h2)
done(f"Found {len(results)} batches")
```

## Docs

- [Positioning](docs/positioning.md) — when to use rlmkit, when not to.
- [Observability](docs/observability.md) — `RLMState`, events, traces,
  sessions, the viewer.
- [Control](docs/control.md) — step loop, checkpoint, fork, rewind,
  intervene, custom prompts / state / tools.
- [Runtimes](docs/runtimes.md) — `Runtime` protocol, Local / Subprocess
  / Docker / Modal, writing your own.
- [Security](docs/security.md) — trust model, Docker isolation knobs,
  approval gates.
- [Changelog](CHANGELOG.md).

## References

- [Recursive Language Models](https://github.com/alexzhang13/rlm) — the
  original RLM paper and implementation.
- [rlm-minimal](https://github.com/alexzhang13/rlm-minimal) — the
  single-file reference rlmkit grew from.
- [ypi](https://github.com/rawwerks/ypi) — recursive coding agent built
  on Pi. Our session layout and much of the default prompt
  (size-up → delegate → combine, guardrails, aggressive delegation) come
  from ypi's `SYSTEM_PROMPT.md`.

## License

See [LICENSE](LICENSE).

## Citation

```bibtex
@misc{sudhakaran2025rlmkit,
  author = {Sudhakaran, Shyam},
  title = {rlmkit},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/shyamsn97/rlmkit}},
}
```
