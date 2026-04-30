"""ReplayLog: append, get, save/load round-trip, fork truncation."""

from __future__ import annotations

from pathlib import Path

import pytest

from rlmflow.workspace import ReplayEntry, ReplayLog


def _entry(branch: str, agent: str, it: int, reply: str = "") -> ReplayEntry:
    return ReplayEntry(
        branch_id=branch,
        agent_id=agent,
        iteration=it,
        reply=reply or f"reply for {agent}:{it}",
        input_tokens=10,
        output_tokens=5,
    )


def test_append_and_get():
    log = ReplayLog()
    e = _entry("b1", "root", 1)
    log.append(e)
    assert log.get("root", 1) is e
    assert log.get("root", 2) is None
    assert log.get("missing", 1) is None


def test_duplicate_append_raises():
    log = ReplayLog()
    log.append(_entry("b1", "root", 1))
    with pytest.raises(ValueError, match="duplicate"):
        log.append(_entry("b2", "root", 1))


def test_save_load_roundtrip(tmp_path: Path):
    path = tmp_path / "replay.jsonl"
    log = ReplayLog(path=path)
    log.append(_entry("b1", "root", 1, "hello"))
    log.append(_entry("b1", "root.child", 1, "world"))
    log.save()

    loaded = ReplayLog.load(path)
    assert len(loaded) == 2
    assert loaded.get("root", 1).reply == "hello"
    assert loaded.get("root.child", 1).reply == "world"
    assert loaded.get("root.child", 1).branch_id == "b1"


def test_load_missing_returns_empty(tmp_path: Path):
    path = tmp_path / "nonexistent.jsonl"
    log = ReplayLog.load(path)
    assert len(log) == 0
    assert log.path == path


def test_fork_drops_target_and_descendants(tmp_path: Path):
    """Fork at root.b:1 drops root.b:1+, root.b.* (descendants)."""
    log = ReplayLog()
    log.append(_entry("b1", "root", 1))           # 0
    log.append(_entry("b1", "root.a", 1))         # 1
    log.append(_entry("b1", "root.b", 1))         # 2  ← target
    log.append(_entry("b1", "root.b.x", 1))       # 3  descendant of target
    log.append(_entry("b1", "root.b", 2))         # 4  later iter of target
    log.append(_entry("b1", "root", 2))           # 5  ancestor, AFTER target

    new_path = tmp_path / "b2.jsonl"
    forked = log.fork(
        agent_id="root.b",
        iteration=1,
        new_branch_id="b2",
        new_path=new_path,
    )

    kept = [(e.agent_id, e.iteration) for e in forked]
    assert ("root", 1) in kept                   # ancestor BEFORE target
    assert ("root.a", 1) in kept                 # sibling
    assert ("root.b", 1) not in kept             # target itself
    assert ("root.b.x", 1) not in kept           # descendant of target
    assert ("root.b", 2) not in kept             # target later iter
    assert ("root", 2) not in kept               # ancestor AFTER target


def test_fork_keeps_independent_siblings_logged_after_target(tmp_path: Path):
    """A sibling that *happened* to log after the target is still kept."""
    log = ReplayLog()
    log.append(_entry("b1", "root", 1))           # 0
    log.append(_entry("b1", "root.b", 1))         # 1  ← target
    log.append(_entry("b1", "root.a", 1))         # 2  sibling, logged AFTER target

    forked = log.fork(
        agent_id="root.b",
        iteration=1,
        new_branch_id="b2",
        new_path=tmp_path / "b2.jsonl",
    )

    kept = [(e.agent_id, e.iteration) for e in forked]
    assert ("root.a", 1) in kept                 # sibling causally independent


def test_fork_preserves_branch_id_provenance(tmp_path: Path):
    """Kept entries keep their original branch_id; new branch_id is for new appends."""
    log = ReplayLog()
    log.append(_entry("b1", "root", 1))
    log.append(_entry("b1", "root.a", 1))
    log.append(_entry("b1", "root.b", 1))         # target

    forked = log.fork(
        agent_id="root.b",
        iteration=1,
        new_branch_id="b2",
        new_path=tmp_path / "b2.jsonl",
    )

    for e in forked:
        assert e.branch_id == "b1"


def test_fork_writes_file(tmp_path: Path):
    log = ReplayLog()
    log.append(_entry("b1", "root", 1))
    log.append(_entry("b1", "root.target", 1))

    new_path = tmp_path / "b2" / "replay.jsonl"
    log.fork(
        agent_id="root.target",
        iteration=1,
        new_branch_id="b2",
        new_path=new_path,
    )

    assert new_path.exists()
    reloaded = ReplayLog.load(new_path)
    assert len(reloaded) == 1
    assert reloaded.get("root", 1) is not None


def test_fork_unknown_target_raises(tmp_path: Path):
    log = ReplayLog()
    log.append(_entry("b1", "root", 1))
    with pytest.raises(ValueError, match="not in replay log"):
        log.fork(
            agent_id="root.missing",
            iteration=1,
            new_branch_id="b2",
            new_path=tmp_path / "b2.jsonl",
        )
