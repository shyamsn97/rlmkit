"""CLI integration tests — path autodetect, render, version, view dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest

from rlmflow import (
    RLM,
    LLMClient,
    LLMUsage,
    RLMConfig,
    RLMNode,
)
from rlmflow.cli import _load, main
from rlmflow.runtime.local import LocalRuntime
from rlmflow.utils.trace import save_trace


class DelegatingLLM(LLMClient):
    """Root delegates once; child finishes immediately."""

    ROOT = (
        "```repl\n"
        "h = delegate('child', 'do the thing')\n"
        "results = yield wait(h)\n"
        "done(results[0])\n"
        "```"
    )
    CHILD = "```repl\ndone('child-answer')\n```"

    def chat(self, messages, *args, **kwargs):
        self.last_usage = LLMUsage(input_tokens=10, output_tokens=5)
        for m in messages:
            if "do the thing" in (m.get("content") or ""):
                return self.CHILD
        return self.ROOT


@pytest.fixture
def run_states() -> list[RLMNode]:
    agent = RLM(
        llm_client=DelegatingLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=2),
    )
    state = agent.start("cli-test")
    states = [state]
    while not state.finished:
        state = agent.step(state)
        states.append(state)
        assert len(states) < 50
    return states


# ── _load() autodetection ────────────────────────────────────────────


def test_load_trace_directory(tmp_path: Path, run_states: list[RLMNode]):
    save_trace(run_states, tmp_path / "run")
    states = _load(tmp_path / "run")
    assert len(states) == len(run_states)
    assert all(isinstance(s, RLMNode) for s in states)


def test_load_trace_file(tmp_path: Path, run_states: list[RLMNode]):
    out = tmp_path / "trace.json"
    save_trace(run_states, out)
    states = _load(out)
    assert len(states) == len(run_states)


def test_load_checkpoint_file(tmp_path: Path, run_states: list[RLMNode]):
    ckpt = tmp_path / "ckpt.json"
    run_states[-1].save(ckpt)
    states = _load(ckpt)
    assert len(states) == 1
    assert states[0].tree(color=False) == run_states[-1].tree(color=False)


def test_load_missing_path(tmp_path: Path):
    with pytest.raises(SystemExit, match="no such file"):
        _load(tmp_path / "nope.json")


def test_load_invalid_json(tmp_path: Path):
    p = tmp_path / "garbage.json"
    p.write_text("{not json")
    with pytest.raises(SystemExit, match="not valid JSON"):
        _load(p)


def test_load_unknown_shape(tmp_path: Path):
    p = tmp_path / "weird.json"
    p.write_text('{"hello": "world"}')
    with pytest.raises(SystemExit, match="doesn't look like"):
        _load(p)


# ── render ───────────────────────────────────────────────────────────


def test_render_mermaid_stdout(
    tmp_path: Path,
    run_states: list[RLMNode],
    capsys: pytest.CaptureFixture,
):
    ckpt = tmp_path / "c.json"
    run_states[-1].save(ckpt)
    rc = main(["render", str(ckpt), "--format", "mermaid"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "stateDiagram-v2" in out
    assert "root" in out


def test_render_dot_to_file(tmp_path: Path, run_states: list[RLMNode]):
    ckpt = tmp_path / "c.json"
    run_states[-1].save(ckpt)
    out_file = tmp_path / "graph.dot"
    rc = main(["render", str(ckpt), "-f", "dot", "-o", str(out_file)])
    assert rc == 0
    text = out_file.read_text()
    assert "digraph" in text
    assert "root" in text


def test_render_tree(
    tmp_path: Path,
    run_states: list[RLMNode],
    capsys: pytest.CaptureFixture,
):
    ckpt = tmp_path / "c.json"
    run_states[-1].save(ckpt)
    rc = main(["render", str(ckpt), "-f", "tree"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "root" in out
    assert "[finished]" in out


def test_render_gantt_html_over_trace(tmp_path: Path, run_states: list[RLMNode]):
    trace_dir = tmp_path / "trace"
    save_trace(run_states, trace_dir)
    out_file = tmp_path / "g.html"
    rc = main(["render", str(trace_dir), "-f", "gantt-html", "-o", str(out_file)])
    assert rc == 0
    text = out_file.read_text()
    assert text.startswith("<!doctype html>")
    assert "gantt" in text.lower()


# ── version ──────────────────────────────────────────────────────────


def test_version(capsys: pytest.CaptureFixture):
    rc = main(["version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "rlmflow" in out
    assert "python" in out


# ── view dispatch (without booting gradio) ───────────────────────────


def test_view_dispatches_to_open_viewer(
    tmp_path: Path,
    run_states: list[RLMNode],
    monkeypatch: pytest.MonkeyPatch,
):
    trace_dir = tmp_path / "trace"
    save_trace(run_states, trace_dir)

    captured: dict = {}

    def fake_open_viewer(states, **kwargs):
        captured["n"] = len(states)
        captured["kwargs"] = kwargs

    monkeypatch.setattr("rlmflow.utils.viewer.open_viewer", fake_open_viewer)

    rc = main(["view", str(trace_dir), "--port", "7861"])
    assert rc == 0
    assert captured["n"] == len(run_states)
    assert captured["kwargs"] == {"server_port": 7861}


# ── argparse surface ─────────────────────────────────────────────────


def test_missing_subcommand_exits():
    with pytest.raises(SystemExit):
        main([])


def test_render_requires_format(tmp_path: Path, run_states: list[RLMNode]):
    ckpt = tmp_path / "c.json"
    run_states[-1].save(ckpt)
    with pytest.raises(SystemExit):
        main(["render", str(ckpt)])
