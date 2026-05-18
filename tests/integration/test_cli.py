"""CLI integration tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util

import pytest

from rlmflow import Graph, LLMClient, LLMUsage, RLMConfig, RLMFlow
from rlmflow.cli import _load, main
from rlmflow.runtime.local import LocalRuntime


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
def run_graphs() -> list[Graph]:
    agent = RLMFlow(
        llm_client=DelegatingLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=2),
    )
    graph = agent.start("cli-test")
    graphs = [graph]
    while not graph.finished:
        graph = agent.step(graph)
        graphs.append(graph)
        assert len(graphs) < 50
    return graphs


def _write_workspace(path: Path, final: Graph) -> Path:
    import json

    path.mkdir()
    manifest = {
        "root_agent_id": final.root_agent_id,
        "agents": list(final.agents),
    }
    (path / "graph.json").write_text(json.dumps(manifest))
    for aid, sub in final.agents.items():
        agent_dir = path / "session" / aid
        agent_dir.mkdir(parents=True)
        (agent_dir / "agent.json").write_text(json.dumps(sub.meta_dict()))
        (agent_dir / "session.jsonl").write_text(
            "\n".join(s.model_dump_json() for s in sub.states) + "\n"
        )
    return path


# ── _load ─────────────────────────────────────────────────────────────


def test_load_graph_file(tmp_path: Path, run_graphs: list[Graph]):
    """A single ``graph.json`` dump becomes a one-element list."""
    ckpt = tmp_path / "graph.json"
    run_graphs[-1].save(ckpt)

    graphs = _load(ckpt)

    assert len(graphs) == 1
    assert graphs[0].tree() == run_graphs[-1].tree()


def test_load_workspace_dir(tmp_path: Path, run_graphs: list[Graph]):
    """A workspace directory with graph.json is loaded as retraced steps."""
    final = run_graphs[-1]
    ws = _write_workspace(tmp_path / "ws", final)

    graphs = _load(ws)
    assert len(graphs) > 1
    assert graphs[-1].result() == final.result()


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

    with pytest.raises(SystemExit, match="does not look like"):
        _load(path)


# ── render ────────────────────────────────────────────────────────────


def test_render_mermaid_stdout(
    tmp_path: Path,
    run_graphs: list[Graph],
    capsys: pytest.CaptureFixture,
):
    ckpt = tmp_path / "g.json"
    run_graphs[-1].save(ckpt)

    rc = main(["render", str(ckpt), "--format", "mermaid"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "stateDiagram-v2" in out
    assert "root" in out


def test_render_dot_to_file(tmp_path: Path, run_graphs: list[Graph]):
    ckpt = tmp_path / "g.json"
    run_graphs[-1].save(ckpt)
    out_file = tmp_path / "graph.dot"

    rc = main(["render", str(ckpt), "-f", "dot", "-o", str(out_file)])

    assert rc == 0
    text = out_file.read_text()
    assert "digraph" in text
    assert "root" in text


def test_render_tree(
    tmp_path: Path,
    run_graphs: list[Graph],
    capsys: pytest.CaptureFixture,
):
    ckpt = tmp_path / "g.json"
    run_graphs[-1].save(ckpt)

    rc = main(["render", str(ckpt), "-f", "tree"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "root" in out
    assert "done -> " in out


def test_render_gantt_html_over_workspace(tmp_path: Path, run_graphs: list[Graph]):
    workspace = _write_workspace(tmp_path / "workspace", run_graphs[-1])
    out_file = tmp_path / "g.html"

    rc = main(["render", str(workspace), "-f", "gantt-html", "-o", str(out_file)])

    assert rc == 0
    text = out_file.read_text()
    assert text.startswith("<!doctype html>")
    assert "gantt" in text.lower()


# ── version ───────────────────────────────────────────────────────────


def test_version(capsys: pytest.CaptureFixture):
    rc = main(["version"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "rlmflow" in out
    assert "python" in out


# ── view dispatch ────────────────────────────────────────────────────


def test_view_dispatches_to_open_viewer(
    tmp_path: Path,
    run_graphs: list[Graph],
    monkeypatch: pytest.MonkeyPatch,
):
    workspace = _write_workspace(tmp_path / "workspace", run_graphs[-1])
    captured: dict = {}

    def fake_open_viewer(graphs, **kwargs):
        captured["n"] = len(graphs)
        captured["kwargs"] = kwargs

    monkeypatch.setattr("rlmflow.utils.viewer.open_viewer", fake_open_viewer)

    rc = main(["view", str(workspace), "--port", "7861"])

    assert rc == 0
    assert captured["n"] > 1
    assert captured["kwargs"] == {"server_port": 7861}


def test_missing_subcommand_exits():
    with pytest.raises(SystemExit):
        main([])


def test_render_requires_format(tmp_path: Path, run_graphs: list[Graph]):
    ckpt = tmp_path / "g.json"
    run_graphs[-1].save(ckpt)

    with pytest.raises(SystemExit):
        main(["render", str(ckpt)])


# ── html / image / steps (kaleido / plotly gated) ────────────────────


@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly not installed")
def test_render_html_writes_stepper(tmp_path: Path, run_graphs: list[Graph]):
    workspace = _write_workspace(tmp_path / "workspace", run_graphs[-1])
    out_file = tmp_path / "stepper.html"

    rc = main(
        [
            "render",
            str(workspace),
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
    assert text.count('<section class="slide') == len(_load(workspace))
    assert "top center" not in text


@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly not installed")
def test_render_html_no_normalize_labels_keeps_top_positions(
    tmp_path: Path, run_graphs: list[Graph]
):
    workspace = _write_workspace(tmp_path / "workspace", run_graphs[-1])
    out_file = tmp_path / "stepper.html"

    rc = main(
        [
            "render",
            str(workspace),
            "-f",
            "html",
            "-o",
            str(out_file),
            "--no-normalize-labels",
        ]
    )
    assert rc == 0
    text = out_file.read_text()
    assert text.startswith("<!doctype html>")
    assert text.count('<section class="slide') == len(_load(workspace))


@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly not installed")
def test_render_html_marker_mult_propagates(
    tmp_path: Path, run_graphs: list[Graph]
):
    workspace = _write_workspace(tmp_path / "workspace", run_graphs[-1])
    out_default = tmp_path / "default.html"
    out_big = tmp_path / "big.html"

    main(["render", str(workspace), "-f", "html", "-o", str(out_default)])
    main(
        [
            "render",
            str(workspace),
            "-f",
            "html",
            "-o",
            str(out_big),
            "--marker-mult",
            "5",
        ]
    )

    assert out_default.read_text() != out_big.read_text()


def test_render_html_requires_out(tmp_path: Path, run_graphs: list[Graph]):
    workspace = _write_workspace(tmp_path / "workspace", run_graphs[-1])

    with pytest.raises(SystemExit, match="--out"):
        main(["render", str(workspace), "-f", "html"])


def test_render_image_requires_out(tmp_path: Path, run_graphs: list[Graph]):
    workspace = _write_workspace(tmp_path / "workspace", run_graphs[-1])

    with pytest.raises(SystemExit, match="--out"):
        main(["render", str(workspace), "-f", "image"])


def test_render_steps_requires_out(tmp_path: Path, run_graphs: list[Graph]):
    workspace = _write_workspace(tmp_path / "workspace", run_graphs[-1])

    with pytest.raises(SystemExit, match="--out"):
        main(["render", str(workspace), "-f", "steps"])


def test_render_image_writes_png(tmp_path: Path, run_graphs: list[Graph]):
    pytest.importorskip("kaleido")
    workspace = _write_workspace(tmp_path / "workspace", run_graphs[-1])
    out_file = tmp_path / "snap.png"

    rc = main(
        [
            "render",
            str(workspace),
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


def test_render_steps_writes_one_per_state(
    tmp_path: Path, run_graphs: list[Graph]
):
    pytest.importorskip("kaleido")
    workspace = _write_workspace(tmp_path / "workspace", run_graphs[-1])
    out_dir = tmp_path / "frames"

    rc = main(
        [
            "render",
            str(workspace),
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
    assert len(files) == len(_load(workspace))


def test_render_steps_split_mults_and_no_normalize(
    tmp_path: Path, run_graphs: list[Graph]
):
    pytest.importorskip("kaleido")
    workspace = _write_workspace(tmp_path / "workspace", run_graphs[-1])
    out_dir = tmp_path / "frames_split"

    rc = main(
        [
            "render",
            str(workspace),
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
    assert len(list(out_dir.glob("step_*.png"))) == len(_load(workspace))
