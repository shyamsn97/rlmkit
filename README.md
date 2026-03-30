# rlmkit

A minimal, pluggable framework for building **recursive LLM agents** in Python.

One agent loop. A Python REPL. The ability to delegate work to sub-agents that get their own fresh context window. That's it.

```
pip install rlmkit
```

## Why

Large tasks don't fit in one context window. `rlmkit` gives your agent a simple recursive pattern: **size up → search → delegate → combine**. Each sub-agent gets a fresh context budget, the same tools, and returns a compact result. No orchestration framework, no DAGs, no YAML — just Python.

## Quick Start

```python
from rlmkit.llm import LLMClient
from rlmkit.logging.rich import RichLogger
from rlmkit.rlm import RLM, RLMConfig
from rlmkit.runtime.local import LocalRuntime


class MyLLM(LLMClient):
    def chat(self, messages):
        # plug in any provider — OpenAI, Anthropic, local, etc.
        ...

runtime = LocalRuntime(workspace=".")
agent = RLM(
    llm_client=MyLLM(),
    runtime=runtime,
    config=RLMConfig(max_depth=3, max_iterations=15),
    logger=RichLogger(),
)
result = agent.run("Find and fix all type errors in src/")
```

## How It Works

```
┌─────────────────────────────────────┐
│  RLM agent loop                     │
│                                     │
│  1. Build system prompt (tools,     │
│     depth, context)                 │
│  2. Send messages to LLM            │
│  3. Extract ```repl``` code block   │
│  4. Execute in persistent namespace │
│  5. Repeat until done() is called   │
│                                     │
│  Tools available in the REPL:       │
│  - done(message)                    │
│  - delegate(task, wait=True/False)  │
│  - wait_all(handles)                │
│  - read_file, write_file, grep, ... │
└─────────────────────────────────────┘
```

When an agent calls `delegate()`, a child agent is created with:
- A **fresh context window** (new message history)
- The **same tools** and configuration
- Its own **runtime** (via `runtime_factory`)
- An incremented **depth** counter

All children share an `AgentPool` (thread pool), so `max_workers` is the global concurrency limit across all depths.

## Core API

### `RLMConfig`

All the tuning knobs, grouped in a dataclass:

```python
RLMConfig(
    max_depth=5,           # recursion limit
    max_iterations=30,     # loops per agent
    max_output_length=12_000,
    replay_last_n_turns=0, # sliding context window
    single_block=True,     # only execute first ```repl``` block
    context_path=None,     # durable scratchpad file
    system_prompt=None,    # raw override (skips default builder)
)
```

### `RLM`

The agent. Subclass and override anything:

| Method | What it does |
|--------|-------------|
| `build_system_prompt()` | Return the system prompt string |
| `call_llm(messages)` | Call the LLM (handles streaming via logger) |
| `execute_code(code)` | Run a REPL block |
| `child_kwargs()` | Control what children inherit |
| `create_child(agent_id, task)` | Full control over child construction |
| `on_run_start/end()` | Lifecycle hooks |
| `on_iteration_start/end()` | Per-iteration hooks |

### `LLMClient`

Abstract base class — implement `chat()`, optionally `stream()`:

```python
class LLMClient(ABC):
    @abstractmethod
    def chat(self, messages: list[dict[str, str]]) -> str: ...

    def stream(self, messages) -> Iterator[str]:
        yield self.chat(messages)  # default fallback
```

### `Runtime`

Execution environment. Only two abstract methods:

```python
class Runtime(ABC):
    @abstractmethod
    def execute(self, code: str, timeout=None) -> str: ...

    @abstractmethod
    def inject(self, name: str, value: Any) -> None: ...
```

`LocalRuntime` runs code via `exec()` in a persistent namespace. File I/O tools (`read_file`, `write_file`, `grep`, `ls`, `edit_file`, `append_file`) are registered as builtins by default.

### Custom Tools

Register tools dynamically with the `@runtime.tool` decorator:

```python
runtime = LocalRuntime(workspace="./data")

@runtime.tool("Search for a regex pattern across files.")
def search(pattern: str, path: str = ".") -> str:
    ...
```

Or use the `@tool` decorator and register manually:

```python
from rlmkit.utils import tool

@tool("Get the current timestamp.")
def now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

runtime.register_tool(now)
```

### Logger

Override any hook for custom output. Three built-in options:

```python
from rlmkit.logging import Logger, PrintLogger
from rlmkit.logging.rich import RichLogger

Logger()       # silent (no-op) — the default
PrintLogger()  # plain stdout streaming
RichLogger()   # syntax-highlighted panels, colored output (requires `rich`)
```

Subclass `Logger` for your own:

```python
class MyLogger(Logger):
    def on_llm_token(self, agent_id, token):
        # stream to a WebSocket, file, TUI, etc.
        ...

    def on_exec_end(self, agent_id, output):
        ...
```

Available hooks: `on_run_start`, `on_run_end`, `on_iter_start`, `on_llm_start`, `on_llm_token`, `on_llm_end`, `on_exec_start`, `on_exec_end`, `on_no_code_block`, `on_stall`, `on_done`, `on_delegate`, `on_delegate_refused`.

### Prompt Builder

Compose markdown prompts from ordered sections:

```python
from rlmkit.prompts import PromptBuilder, Section

builder = PromptBuilder(
    order="{role}\n\n{instructions}\n\n{tools}",
    sections={
        "role": Section("role", "You are a helpful agent.", title="Role"),
        "instructions": "Be concise.",
    },
)
builder.section("tools", lambda ctx: ctx["tool_list"], title="Tools")
prompt = builder.build(context={"tool_list": "- read_file\n- grep"})
```

## Examples

### `examples/basic.py` — Single-file needle in a haystack

Generates a 1M-line file with a hidden magic number. The agent chunks the file and delegates search to sub-agents in parallel. Demonstrates dynamically registered tools on a bare runtime.

### `examples/needle_haystack.py` — Multi-file needle search

Generates 500 files with a needle in one. Uses `@runtime.tool` for custom `grep`, `list_files`, `count_files`, and `read_file`. Passes a `runtime_factory` so each child agent gets a fresh runtime with the same tools.

### `examples/custom_agent.py` — Subclassing RLM

A `CodeReviewer` agent that overrides `build_system_prompt()`, registers custom tools (`file_stats`, `now`), uses lifecycle hooks for timing, and caps child iterations in `create_child()`.

## Project Structure

```
rlmkit/
├── llm.py                  # LLMClient ABC
├── rlm.py                  # RLM, RLMConfig, AgentPool, DelegateHandle
├── utils.py                # @tool decorator, code block parsing
├── logging/
│   ├── base.py             # Logger (no-op), PrintLogger
│   └── rich.py             # RichLogger
├── runtime/
│   ├── runtime.py          # Runtime ABC, ToolDef, file I/O builtins
│   └── local.py            # LocalRuntime (exec-based)
└── prompts/
    ├── builder.py          # PromptBuilder, Section
    └── default.py          # Default prompt text + sections
```

## License

See [LICENSE](LICENSE).
