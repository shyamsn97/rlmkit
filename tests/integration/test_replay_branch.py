"""End-to-end record / replay / fork integration test.

Uses a scripted LLM so the test is deterministic offline. Verifies:

1. Recording: a real run populates the replay log with entries.
2. Strict replay: a second run with a fail-fast LLM produces the same
   final result and makes ZERO live LLM calls.
3. Fork: ``RLMNode.fork(...)`` copies workspace/session and carries
   the replay plan. The forked run replays the prefix from the node
   graph and only burns live calls on the divergent tail.
"""

from __future__ import annotations

from pathlib import Path

from rlmkit import RLM, RLMConfig, ReplayLog, Workspace
from rlmkit.llm import LLMClient, LLMUsage


def _run_node(engine: RLM, node):
    while not node.finished:
        node = engine.step(node)
    return node


# Root delegates two children, then summarizes their results.
ROOT_REPLY = """```repl
h1 = delegate("worker_a", "task A")
h2 = delegate("worker_b", "task B")
results = yield wait(h1, h2)
done(results[0] + "+" + results[1])
```"""


class ScriptedLLM(LLMClient):
    """Replies based on the query in the user messages.

    Counts total live calls so we can verify caching behavior.
    """

    def __init__(self, tag: str = "default") -> None:
        self.tag = tag
        self.call_count = 0

    def chat(self, messages, *args, **kwargs):
        self.call_count += 1
        self.last_usage = LLMUsage(input_tokens=100, output_tokens=20)

        # Identify the agent by its task text in the first user message.
        first_user = next((m for m in messages if m["role"] == "user"), None)
        text = (first_user or {}).get("content", "")

        if "task A" in text:
            return f'```repl\ndone("{self.tag}-A")\n```'
        if "task B" in text:
            return f'```repl\ndone("{self.tag}-B")\n```'
        # Otherwise: root. Issue the delegate-and-wait dance.
        return ROOT_REPLY


class FailIfCalledLLM(LLMClient):
    """Bombs on any live call. Used to assert pure replay."""

    def chat(self, messages, *args, **kwargs):  # pragma: no cover
        raise AssertionError("LLM was called during strict replay")


def _make_engine(
    llm: LLMClient,
    run_dir: Path,
    *,
    branch_id: str = "main",
) -> RLM:
    workspace = Workspace.create(run_dir, branch_id=branch_id)
    return RLM(
        llm_client=llm,
        workspace=workspace,
        config=RLMConfig(
            max_depth=1,
            max_iterations=10,
        ),
    )


def _fork_engine(
    source,
    llm: LLMClient,
    tmp_path: Path,
    *,
    instruction: str | None = None,
) -> tuple[RLM, object]:
    forked = source.fork(
        "root.worker_a",
        branch_id="b2",
        workspace=tmp_path / "b2",
        iteration=1,
        instruction=instruction,
    )
    engine = RLM(
        llm_client=llm,
        workspace=Workspace.open(forked.workspace),
        config=RLMConfig(max_depth=1, max_iterations=10),
    )
    return engine, forked


def test_records_and_replays_strictly(tmp_path: Path):
    b1_dir = tmp_path / "b1"
    live_llm = ScriptedLLM(tag="b1")

    engine = _make_engine(live_llm, b1_dir, branch_id="b1")
    result = engine.run("test query")

    assert result == "b1-A+b1-B"
    # 1 root + 2 workers = 3 LLM calls
    log = Workspace.create(b1_dir, branch_id="b1").materialize_replay_log()
    assert len(log) == 3
    by_agent = sorted((e.agent_id, e.iteration) for e in log)
    assert by_agent == [("root", 1), ("root.worker_a", 1), ("root.worker_b", 1)]
    assert all(e.branch_id == "b1" for e in log)

    # Strict replay: must NOT make a live call.
    fail_llm = FailIfCalledLLM()
    replay_engine = _make_engine(
        fail_llm,
        b1_dir,
        branch_id="b1",
    )
    replay_result = replay_engine.run("test query")
    assert replay_result == "b1-A+b1-B"


def test_fork_replays_prefix_only_runs_divergent_tail(tmp_path: Path):
    b1_dir = tmp_path / "b1"

    # 1. Record b1.
    b1_llm = ScriptedLLM(tag="b1")
    engine = _make_engine(b1_llm, b1_dir, branch_id="b1")
    b1_node = _run_node(engine, engine.start("test query"))
    b1_result = b1_node.result
    assert b1_result == "b1-A+b1-B"
    assert b1_llm.call_count == 3

    # 2. Fork at root.worker_a:1, swapping in a b2-tagged LLM for the tail.
    b2_llm = ScriptedLLM(tag="b2")
    fork_engine, forked_node = _fork_engine(b1_node, b2_llm, tmp_path)

    # Replay prefix is carried by the forked node, not an engine-level fork.
    assert forked_node.replay is not None
    assert forked_node.replay.target_id == "root.worker_a"

    # 3. Run b2 — only worker_a should hit the live LLM.
    b2_node = _run_node(fork_engine, forked_node)
    b2_result = b2_node.result

    # b2's live LLM should have been called exactly ONCE — for worker_a.
    # (root iter 1 and worker_b iter 1 replay from the source node graph.)
    assert b2_llm.call_count == 1

    # The result composes b2's fresh A-result with b1's cached B-result.
    assert b2_result == "b2-A+b1-B"

    # 4. Fork log records only fresh branch calls; source replay lives on the node.
    final_log = ReplayLog.load(tmp_path / "b2" / "replay.jsonl")
    assert len(final_log) == 1
    branches = {e.branch_id for e in final_log}
    assert branches == {"b2"}
    new_entry = final_log.get("root.worker_a", 1)
    assert new_entry is not None
    assert new_entry.branch_id == "b2"


def test_fork_instruction_is_visible_to_live_branch_and_children(tmp_path: Path):
    b1_dir = tmp_path / "b1"
    engine = _make_engine(ScriptedLLM(tag="b1"), b1_dir, branch_id="b1")
    b1_node = _run_node(engine, engine.start("test query"))

    fork_engine, forked_node = _fork_engine(
        b1_node,
        ScriptedLLM(tag="b2"),
        tmp_path,
        instruction="retry worker_a with a different strategy",
    )

    prompt = fork_engine.build_system_prompt(forked_node)
    assert "## Branch Instruction" in prompt
    assert "retry worker_a with a different strategy" in prompt


def test_fork_isolates_session_writes(tmp_path: Path):
    b1_dir = tmp_path / "b1"
    b1_llm = ScriptedLLM(tag="b1")
    engine = _make_engine(b1_llm, b1_dir, branch_id="b1")
    engine.run("test query")

    b1_session_root = (b1_dir / "session" / "session.json").read_text()
    assert "b1-A" in b1_session_root or "worker_a" in b1_session_root

    fork_engine, forked_node = _fork_engine(
        engine.last_node,
        ScriptedLLM(tag="b2"),
        tmp_path,
    )
    _run_node(fork_engine, forked_node)

    # b1's session must be byte-for-byte unchanged after b2 ran.
    assert (b1_dir / "session" / "session.json").read_text() == b1_session_root
