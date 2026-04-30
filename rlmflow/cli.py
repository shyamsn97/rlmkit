"""``rlmflow`` command-line entry point.

Three sub-commands, all operating on paths — no agent construction.

    rlmflow view     <path>              open the Gradio viewer
    rlmflow render   <path> --format F   write a static render
    rlmflow version                      print package + environment info

Dispatch is plain ``argparse``; there are no optional third-party
dependencies. ``rlmflow view`` still needs ``gradio`` at run-time.

``main`` is importable (``from rlmflow.cli import main``) so callers can
wrap or alias the CLI in their own entry points.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rlmflow.node import Node

# ── path autodetect ──────────────────────────────────────────────────


def _load(path: Path) -> list[Node]:
    """Return a list of states for *path* — trace, checkpoint, or dir."""
    from rlmflow.utils.trace import load_trace

    if path.is_dir():
        return _load(path / "trace.json")

    if not path.is_file():
        raise SystemExit(f"rlmflow: no such file or directory: {path}")

    try:
        head = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"rlmflow: {path} is not valid JSON: {e}") from None

    if isinstance(head, dict) and "steps" in head:
        return load_trace(path).states
    if isinstance(head, dict) and "agent_id" in head:
        return [Node.load(path)]
    raise SystemExit(f"rlmflow: {path} doesn't look like a trace or a state checkpoint")


# ── commands ─────────────────────────────────────────────────────────


def cmd_view(args: argparse.Namespace) -> int:
    from rlmflow.utils.viewer import open_viewer

    states = _load(Path(args.path))
    launch_kwargs: dict[str, Any] = {}
    if args.share:
        launch_kwargs["share"] = True
    if args.port is not None:
        launch_kwargs["server_port"] = args.port
    if args.host is not None:
        launch_kwargs["server_name"] = args.host
    open_viewer(states, **launch_kwargs)
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    from rlmflow.utils.export import to_dot, to_mermaid
    from rlmflow.utils.viz import gantt_html

    states = _load(Path(args.path))
    final = states[-1]

    if args.format == "mermaid":
        out = to_mermaid(final)
    elif args.format == "dot":
        out = to_dot(final)
    elif args.format == "tree":
        out = final.tree(color=False)
    elif args.format == "gantt-html":
        out = gantt_html(states)
    else:
        raise SystemExit(f"rlmflow: unknown format {args.format!r}")

    if args.out:
        Path(args.out).write_text(out)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(out)
        if not out.endswith("\n"):
            sys.stdout.write("\n")
    return 0


def cmd_version(_args: argparse.Namespace) -> int:
    import platform

    try:
        from importlib.metadata import version as _pkg_version

        pkg = _pkg_version("rlmflow")
    except Exception:
        pkg = "unknown"

    try:
        import gradio  # noqa: F401

        gradio_status = "available"
    except ImportError:
        gradio_status = "not installed"

    print(f"rlmflow  {pkg}")
    print(f"python  {platform.python_version()} ({sys.platform})")
    print(f"gradio  {gradio_status}")
    return 0


# ── parser ───────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rlmflow",
        description="rlmflow command-line tools",
    )
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

    v = sub.add_parser(
        "view",
        help="open the Gradio viewer on a trace or state checkpoint",
    )
    v.add_argument(
        "path",
        help="trace directory, trace.json, or single state checkpoint",
    )
    v.add_argument("--share", action="store_true", help="create a public URL")
    v.add_argument("--port", type=int, default=None, help="server port")
    v.add_argument("--host", default=None, help="server host / bind address")
    v.set_defaults(func=cmd_view)

    r = sub.add_parser(
        "render",
        help="render a trace or state as mermaid / dot / tree / gantt-html",
    )
    r.add_argument("path", help="trace directory, trace.json, or checkpoint")
    r.add_argument(
        "--format",
        "-f",
        required=True,
        choices=["mermaid", "dot", "tree", "gantt-html"],
        help="output format",
    )
    r.add_argument(
        "--out",
        "-o",
        default=None,
        help="write to file (default: stdout)",
    )
    r.set_defaults(func=cmd_render)

    ver = sub.add_parser("version", help="print package and environment info")
    ver.set_defaults(func=cmd_version)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
