"""CLI integration tests for typed RLMFlow node traces."""

from __future__ import annotations

from pathlib import Path
import importlib.util

import pytest

from rlmflow import LLMClient, LLMUsage, Node, RLMConfig, RLMFlow
from rlmflow.cli import _load, main
from rlmflow.runtime.local import LocalRuntime
from rlmflow.utils.trace import save_trace


PLOTLY_INSTALLED = importlib.util.find_spec("plotly") is not None


class DelegatingLLM(LLMClient):
    ROOT = (
        "```repl\n"
        "h = delegate('child', 'do the thing', '')\n"
        "results = yield wait(h)\n"
        "done(results[0])\n"
        "```"
    )
    CHILD = "```repl\ndone('child-answer')\n```"

    def chat(self, messages, *args, **kwargs):
        self.last_usage = LLMUsage(input_tokens=10, output_tokens=5)
        for message in messages:
            if "do the thing" in (message.get("content") or ""):
                return self.CHILD
        return self.ROOT


@pytest.fixture
def run_states() -> list[Node]:
    agent = RLMFlow(
        llm_client=DelegatingLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=2),
    )
    node = agent.start("cli-test")
    states = [node]
    while not node.finished:
        node = agent.step(node)
        states.append(node)
        assert len(states) < 50
    return states


def test_load_trace_directory(tmp_path: Path, run_states: list[Node]):
    save_trace(run_states, tmp_path / "run")

    states = _load(tmp_path / "run")

    assert len(states) == len(run_states)
    assert all(isinstance(state, Node) for state in states)


def test_load_trace_file(tmp_path: Path, run_states: list[Node]):
    out = tmp_path / "trace.json"
    save_trace(run_states, out)

    assert len(_load(out)) == len(run_states)


def test_load_checkpoint_file(tmp_path: Path, run_states: list[Node]):
    ckpt = tmp_path / "ckpt.json"
    run_states[-1].save(ckpt)

    states = _load(ckpt)

    assert len(states) == 1
    assert states[0].tree(color=False) == run_states[-1].tree(color=False)


def test_load_missing_path(tmp_path: Path):
    with pytest.raises(SystemExit, match="no such file"):
        _load(tmp_path / "nope.json")


def test_load_invalid_json(tmp_path: Path):
    path = tmp_path / "garbage.json"
    path.write_text("{not json")

    with pytest.raises(SystemExit, match="not valid JSON"):
        _load(path)


def test_load_unknown_shape(tmp_path: Path):
    path = tmp_path / "weird.json"
    path.write_text('{"hello": "world"}')

    with pytest.raises(SystemExit, match="doesn't look like"):
        _load(path)


def test_render_mermaid_stdout(
    tmp_path: Path,
    run_states: list[Node],
    capsys: pytest.CaptureFixture,
):
    ckpt = tmp_path / "c.json"
    run_states[-1].save(ckpt)

    rc = main(["render", str(ckpt), "--format", "mermaid"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "stateDiagram-v2" in out
    assert "root" in out


def test_render_dot_to_file(tmp_path: Path, run_states: list[Node]):
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
    run_states: list[Node],
    capsys: pytest.CaptureFixture,
):
    ckpt = tmp_path / "c.json"
    run_states[-1].save(ckpt)

    rc = main(["render", str(ckpt), "-f", "tree"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "root" in out
    assert "[result]" in out


def test_render_gantt_html_over_trace(tmp_path: Path, run_states: list[Node]):
    trace_dir = tmp_path / "trace"
    save_trace(run_states, trace_dir)
    out_file = tmp_path / "g.html"

    rc = main(["render", str(trace_dir), "-f", "gantt-html", "-o", str(out_file)])

    assert rc == 0
    text = out_file.read_text()
    assert text.startswith("<!doctype html>")
    assert "gantt" in text.lower()


def test_version(capsys: pytest.CaptureFixture):
    rc = main(["version"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "rlmflow" in out
    assert "python" in out


def test_view_dispatches_to_open_viewer(
    tmp_path: Path,
    run_states: list[Node],
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


def test_missing_subcommand_exits():
    with pytest.raises(SystemExit):
        main([])


def test_render_requires_format(tmp_path: Path, run_states: list[Node]):
    ckpt = tmp_path / "c.json"
    run_states[-1].save(ckpt)

    with pytest.raises(SystemExit):
        main(["render", str(ckpt)])


@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly not installed")
def test_render_html_writes_stepper(tmp_path: Path, run_states: list[Node]):
    trace_dir = tmp_path / "trace"
    save_trace(run_states, trace_dir)
    out_file = tmp_path / "stepper.html"

    rc = main(
        [
            "render",
            str(trace_dir),
            "-f",
            "html",
            "-o",
            str(out_file),
            "--title",
            "cli stepper",
        ]
    )

    assert rc == 0
    text = out_file.read_text()
    assert text.startswith("<!doctype html>")
    assert "<title>cli stepper</title>" in text
    # one slide per state
    assert text.count('<section class="slide') == len(run_states)
    # default normalize_labels=True should erase every "top *" position
    # in the embedded Plotly JSON — same baseline as save_image / save_steps.
    assert "top center" not in text


@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly not installed")
def test_render_html_no_normalize_labels_keeps_top_positions(
    tmp_path: Path, run_states: list[Node]
):
    """--no-normalize-labels lets the alternating layout through, matching
    the live Gradio viewer."""
    trace_dir = tmp_path / "trace"
    save_trace(run_states, trace_dir)
    out_file = tmp_path / "stepper.html"

    rc = main(
        [
            "render",
            str(trace_dir),
            "-f",
            "html",
            "-o",
            str(out_file),
            "--no-normalize-labels",
        ]
    )
    assert rc == 0
    text = out_file.read_text()
    assert "top center" in text


@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly not installed")
def test_render_html_marker_mult_propagates(
    tmp_path: Path, run_states: list[Node]
):
    """--marker-mult should reach the embedded Plotly JSON via render_html."""
    trace_dir = tmp_path / "trace"
    save_trace(run_states, trace_dir)
    out_default = tmp_path / "default.html"
    out_big = tmp_path / "big.html"

    main(["render", str(trace_dir), "-f", "html", "-o", str(out_default)])
    main(
        [
            "render",
            str(trace_dir),
            "-f",
            "html",
            "-o",
            str(out_big),
            "--marker-mult",
            "5",
        ]
    )

    assert out_default.read_text() != out_big.read_text(), (
        "--marker-mult should change the rendered HTML; if this fails the "
        "scaling didn't propagate through node_plot_html → render_html."
    )


def test_render_html_requires_out(tmp_path: Path, run_states: list[Node]):
    trace_dir = tmp_path / "trace"
    save_trace(run_states, trace_dir)

    with pytest.raises(SystemExit, match="--out"):
        main(["render", str(trace_dir), "-f", "html"])


def test_render_image_requires_out(tmp_path: Path, run_states: list[Node]):
    trace_dir = tmp_path / "trace"
    save_trace(run_states, trace_dir)

    with pytest.raises(SystemExit, match="--out"):
        main(["render", str(trace_dir), "-f", "image"])


def test_render_steps_requires_out(tmp_path: Path, run_states: list[Node]):
    trace_dir = tmp_path / "trace"
    save_trace(run_states, trace_dir)

    with pytest.raises(SystemExit, match="--out"):
        main(["render", str(trace_dir), "-f", "steps"])


def test_render_image_writes_png(tmp_path: Path, run_states: list[Node]):
    pytest.importorskip("kaleido")
    trace_dir = tmp_path / "trace"
    save_trace(run_states, trace_dir)
    out_file = tmp_path / "snap.png"

    rc = main(
        [
            "render",
            str(trace_dir),
            "-f",
            "image",
            "-o",
            str(out_file),
            "--width",
            "300",
            "--height",
            "240",
            "--scale",
            "1.0",
            "--element-mult",
            "1.5",
        ]
    )

    assert rc == 0
    assert out_file.exists()
    with open(out_file, "rb") as fh:
        assert fh.read(4) == b"\x89PNG"


def test_render_steps_writes_one_per_state(tmp_path: Path, run_states: list[Node]):
    pytest.importorskip("kaleido")
    trace_dir = tmp_path / "trace"
    save_trace(run_states, trace_dir)
    out_dir = tmp_path / "frames"

    rc = main(
        [
            "render",
            str(trace_dir),
            "-f",
            "steps",
            "-o",
            str(out_dir),
            "--width",
            "300",
            "--height",
            "240",
            "--scale",
            "1.0",
            "--element-mult",
            "1.5",
        ]
    )

    assert rc == 0
    files = sorted(p.name for p in out_dir.glob("step_*.png"))
    assert len(files) == len(run_states)


def test_render_steps_split_mults_and_no_normalize(
    tmp_path: Path, run_states: list[Node]
):
    """Split marker / text mults and --no-normalize-labels both flow through."""
    pytest.importorskip("kaleido")
    trace_dir = tmp_path / "trace"
    save_trace(run_states, trace_dir)
    out_dir = tmp_path / "frames_split"

    rc = main(
        [
            "render",
            str(trace_dir),
            "-f",
            "steps",
            "-o",
            str(out_dir),
            "--width",
            "320",
            "--height",
            "240",
            "--scale",
            "1.0",
            "--marker-mult",
            "3.5",
            "--text-mult",
            "2.2",
            "--no-normalize-labels",
        ]
    )

    assert rc == 0
    assert len(list(out_dir.glob("step_*.png"))) == len(run_states)
