# When to use rlmkit

A guide for choosing between rlmkit and other agent frameworks.

## The one-line pitch

rlmkit is a **state machine for recursive LLM agents**. Every agent in the tree advances one step at a time, and the entire computation is a single frozen, serializable object you can inspect, checkpoint, fork, or replay.

## Decision matrix

| You want to... | Use |
|----------------|-----|
| Build a recursive agent where sub-agents spawn sub-agents | **rlmkit** |
| Checkpoint/fork/rewind agent execution mid-run | **rlmkit** |
| Run a one-off coding task from your terminal | Codex CLI, Claude Code, Aider |
| Build a graph of heterogeneous agents with complex routing | LangGraph |
| Assemble a team of role-based agents (researcher + writer + reviewer) | CrewAI |
| Optimize prompts/programs against a metric | DSPy |
| Run SWE-bench evaluations with minimal setup | SWE-agent, OpenHands |
| Pair-program with git integration and auto-commits | Aider |

## rlmkit vs specific alternatives

### vs rlm-minimal

[rlm-minimal](https://github.com/alexzhang13/rlm-minimal) is the reference implementation rlmkit grew from. It proves the core RLM idea in a single file.

**Choose rlm-minimal** when you want the simplest possible recursive agent to study or fork.

**Choose rlmkit** when you need:
- Step-level control (pause, inspect, resume)
- Immutable state with checkpoint/fork/time-travel
- Parallel child execution with a thread pool
- Session persistence
- Interactive visualization (Gradio viewer)
- Token tracking and budgets
- Multiple LLM backends and model routing

### vs ypi

[ypi](https://github.com/rawwerks/ypi) is a shell-first recursive agent built on Pi. We borrowed prompt patterns and session design from it.

**Choose ypi** when you want a polished CLI coding agent with jj workspace isolation per child.

**Choose rlmkit** when you want an **embeddable Python library** with:
- Typed config and Pydantic state (not env vars and subprocess orchestration)
- First-class immutable tree state you can serialize/replay
- Fine-grained stepping visible to your own code
- Custom runtimes, tools, and prompt builders

### vs LangGraph

[LangGraph](https://github.com/langchain-ai/langgraph) is a graph DSL with channel-based state, durable checkpointing (SQLite/Postgres), and a huge ecosystem.

**Choose LangGraph** when you need:
- Complex graph topologies beyond recursive trees
- Production-grade durable checkpointing to databases
- Large ecosystem of prebuilt integrations
- Enterprise features (LangSmith tracing, deployment)

**Choose rlmkit** when:
- Your problem is naturally recursive (one agent type, spawning copies of itself)
- You want the mental model to be "LLM + REPL + delegation" not "graph nodes + channels"
- You want a single frozen tree as your state primitive
- You prefer minimal dependencies (~1,200 lines of core code)

### vs CrewAI

[CrewAI](https://github.com/crewAIInc/crewAI) builds teams of role-based agents with Flows for deterministic orchestration.

**Choose CrewAI** when you want pre-defined roles (researcher, writer, reviewer) collaborating on a task, with MCP and enterprise tracing.

**Choose rlmkit** when your agents are homogeneous (same prompt, different subtask) and the tree structure emerges from the problem, not from hand-designed roles.

### vs AutoGen

[AutoGen](https://github.com/microsoft/autogen) is an actor-based multi-agent framework with event-driven messaging.

**Choose AutoGen** when you need async multi-agent messaging at scale, OpenTelemetry, or Microsoft ecosystem integration.

**Choose rlmkit** when you want a simpler synchronous step API where the entire tree is one inspectable object.

### vs SWE-agent / OpenHands

These are task-specific harnesses for coding agents with Docker sandboxes and benchmark infrastructure.

**Choose SWE-agent/OpenHands** when you're running SWE-bench or need a ready-made coding harness.

**Choose rlmkit** when you're building a general recursive agent (not just code patching) and want full control over the execution loop.

### vs Aider

[Aider](https://github.com/paul-gauthier/aider) is a git-first pair programmer — the best single-agent coding UX.

**Choose Aider** for daily coding tasks with automatic git commits and rollbacks.

**Choose rlmkit** when you need agents to recursively decompose and parallelize work across a tree.

## What rlmkit is NOT

- **Not a product** — it's a library. You bring your own LLM keys, runtime, and UI.
- **Not a sandbox** — `LocalRuntime` runs code in your process. Use `ModalRuntime` or a custom `Runtime` for isolation.
- **Not an integration platform** — we don't ship 100 tool connectors. Register your own tools via `@runtime.tool` or use the built-in filesystem tools.
- **Not a prompt optimizer** — we provide sensible defaults, but prompt tuning is your job (or pair with DSPy).

## Who should use rlmkit

- **Researchers** studying recursive/hierarchical agent architectures who want inspectable, replayable execution traces.
- **Engineers** building agents that naturally decompose work (code gen, search, summarization) and want step-level control.
- **Teams** that need checkpoint/fork/replay for debugging complex multi-agent runs.
- **RL practitioners** who want a gym-style `step()` loop over agent execution.
