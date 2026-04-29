"""Integration test: step-level control primitives.

Exercises the operations promised in ``docs/control.md``:
- step-by-step loop
- ``state.model_dump_json`` → ``model_validate_json`` round trip
- list-of-states rewind / fork (divergence after a branch)
- intervention via ``state.update(...)`` killing a child mid-run
"""

from __future__ import annotations

from rlmkit import (
    RLM,
    LLMClient,
    LLMUsage,
    RLMConfig,
    RLMNode,
    Status,
)
from rlmkit.runtime.local import LocalRuntime


class ScriptedLLM(LLMClient):
    """Returns replies from a per-agent script; raises if exhausted.

    Keyed by ``agent_id`` so the same LLM instance can drive the root and
    any number of children deterministically.
    """

    def __init__(self, scripts: dict[str, list[str]]) -> None:
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.cursors: dict[str, int] = {k: 0 for k in scripts}

    def chat(self, messages, *args, **kwargs):
        self.last_usage = LLMUsage(input_tokens=4, output_tokens=4)
        aid = self._agent_id(messages)
        script = self.scripts.get(aid)
        if not script:
            raise AssertionError(f"no script for {aid!r}")
        i = self.cursors[aid]
        if i >= len(script):
            raise AssertionError(f"script exhausted for {aid!r}")
        self.cursors[aid] = i + 1
        return script[i]

    @staticmethod
    def _agent_id(messages: list[dict]) -> str:
        sysmsg = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""
        for line in sysmsg.splitlines():
            if line.startswith("AGENT_ID"):
                return line.split("=", 1)[1].strip()
        # Fallback: find the 'do the thing' child query or default to root.
        for m in messages:
            c = m.get("content") or ""
            if "delegate-done" in c:
                return "root.child"
        return "root"


def _done(text: str) -> str:
    return f"```repl\ndone({text!r})\n```"


def _delegate_one(child_name: str, query: str) -> str:
    return (
        "```repl\n"
        f"h = delegate({child_name!r}, {query!r})\n"
        "results = yield wait(h)\n"
        "done(results[0])\n"
        "```"
    )


# ── tests ─────────────────────────────────────────────────────────────


def test_step_loop_reaches_finished():
    llm = ScriptedLLM({"root": [_done("direct-answer")]})
    agent = RLM(llm_client=llm, runtime=LocalRuntime())

    state = agent.start("control-step")
    states = [state]
    while not state.finished:
        state = agent.step(state)
        states.append(state)
        assert len(states) < 20

    assert state.status == Status.FINISHED
    assert state.result == "direct-answer"


def test_checkpoint_round_trip_resumes_cleanly():
    llm = ScriptedLLM({"root": [_done("checkpoint-me")]})
    agent = RLM(llm_client=llm, runtime=LocalRuntime())

    state = agent.start("control-ckpt")
    while not state.finished:
        state = agent.step(state)

    blob = state.model_dump_json()
    restored = RLMNode.model_validate_json(blob)

    assert restored.status == state.status == Status.FINISHED
    assert restored.result == state.result
    assert restored.messages == state.messages
    assert restored.tree_usage() == state.tree_usage()


def test_fork_from_midrun_state_diverges():
    """Two agents sharing a midrun snapshot can produce different results."""
    llm_a = ScriptedLLM({"root": [_done("path-A")]})
    llm_b = ScriptedLLM({"root": [_done("path-B")]})

    agent_a = RLM(llm_client=llm_a, runtime=LocalRuntime())
    agent_b = RLM(llm_client=llm_b, runtime=LocalRuntime())

    mid = agent_a.start("control-fork")

    final_a = mid
    while not final_a.finished:
        final_a = agent_a.step(final_a)

    resumed = agent_b.restore(mid)
    while not resumed.finished:
        resumed = agent_b.step(resumed)

    assert final_a.result == "path-A"
    assert resumed.result == "path-B"


def test_rewind_is_just_list_indexing():
    llm = ScriptedLLM({"root": [_done("rewind-me")]})
    agent = RLM(llm_client=llm, runtime=LocalRuntime())

    state = agent.start("control-rewind")
    states = [state]
    while not state.finished:
        state = agent.step(state)
        states.append(state)

    earlier = states[0]
    latest = states[-1]
    assert earlier.status == Status.READY
    assert latest.status == Status.FINISHED
    # Earlier snapshot is unchanged by later stepping — immutability.
    assert earlier is not latest
    assert earlier.iteration == 0
    assert latest.iteration >= 1


def test_intervene_kills_a_child_mid_run():
    """Force-finish a child by rewriting the supervising state between steps."""
    llm = ScriptedLLM(
        {
            "root": [_delegate_one("child", "delegate-done"), _done("root-done")],
            "root.child": [_done("(will not finish)")],
        }
    )
    agent = RLM(
        llm_client=llm,
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=2),
    )

    state = agent.start("control-intervene")
    while state.status != Status.SUPERVISING:
        state = agent.step(state)
        assert state.iteration < 20

    assert state.waiting_on == ["root.child"]

    # Intervene: mark the child finished with a canned result and clear the wait.
    killed_child = state.children[0].update(
        status=Status.FINISHED,
        result="killed-by-supervisor",
    )
    patched = state.update(children=[killed_child])
    # Mirror the intervention in the engine's child registry so resume_exec
    # pulls the canned result.
    agent.child_engines["root.child"].is_done = True
    agent.child_engines["root.child"].result = "killed-by-supervisor"

    while not patched.finished:
        patched = agent.step(patched)
        assert patched.iteration < 20

    assert patched.status == Status.FINISHED
    assert patched.children[0].result == "killed-by-supervisor"
