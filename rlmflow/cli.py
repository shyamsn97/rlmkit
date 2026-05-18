"""``rlmflow`` command-line entry point.

Three sub-commands, all operating on paths — no agent construction.

    rlmflow view     <path>              open the Gradio viewer
    rlmflow render   <path> --format F   write a static render
    rlmflow version                      print package + environment info

``<path>`` may be:

* a workspace directory (``graph.json`` + ``session/`` + ``context/``)
  — retraced into graph snapshots from the persisted session.
* a standalone ``Graph`` JSON snapshot.

``--format`` accepts text formats (``mermaid`` / ``dot`` / ``d2`` /
``tree`` / ``report-md`` / ...) and binary/viz formats (``html`` for a
self-contained stepper, ``image`` for a single PNG/SVG, ``steps`` for
one image per snapshot under ``--out`` directory).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from rlmflow.graph import Graph


def _load(path: Path) -> list[Graph]:
    """Return graph snapshots for a workspace, graph dir, or graph dump."""
    from rlmflow.utils.viewer import resolve_graphs

    try:
        return resolve_graphs(path)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"rlmflow: {exc}") from None


# ── commands ─────────────────────────────────────────────────────────


def cmd_view(args: argparse.Namespace) -> int:
    from rlmflow.utils.viewer import open_viewer

    graphs = _load(Path(args.path))
    launch_kwargs: dict[str, Any] = {}
    if args.share:
        launch_kwargs["share"] = True
    if args.port is not None:
        launch_kwargs["server_port"] = args.port
    if args.host is not None:
        launch_kwargs["server_name"] = args.host
    open_viewer(graphs, **launch_kwargs)
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    from rlmflow.utils.export import (
        to_d2,
        to_dot,
        to_mermaid,
        to_mermaid_flowchart,
        to_mermaid_sequence,
    )
    from rlmflow.utils.viz import (
        ascii_boxes,
        code_log,
        error_summary,
        gantt_html,
        report_md,
        token_sparkline,
    )

    graphs = _load(Path(args.path))
    topo = graphs[-1]

    fmt = args.format
    if fmt in ("html", "image", "steps"):
        return _render_viz(args, graphs, topo, fmt)

    if fmt == "mermaid":
        out = to_mermaid(topo)
    elif fmt == "mermaid-flowchart":
        out = to_mermaid_flowchart(topo)
    elif fmt == "mermaid-sequence":
        out = to_mermaid_sequence(topo)
    elif fmt == "dot":
        out = to_dot(topo)
    elif fmt == "d2":
        out = to_d2(topo)
    elif fmt == "tree":
        out = topo.tree()
    elif fmt == "ascii-boxes":
        out = ascii_boxes(topo)
    elif fmt == "gantt-html":
        out = gantt_html(graphs)
    elif fmt == "report-md":
        out = report_md(graphs)
    elif fmt == "code-log":
        out = code_log(topo)
    elif fmt == "error-summary":
        out = error_summary(topo)
    elif fmt == "tokens":
        out = token_sparkline(graphs)
    else:
        raise SystemExit(f"rlmflow: unknown format {fmt!r}")

    if args.out:
        Path(args.out).write_text(out)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(out)
        if not out.endswith("\n"):
            sys.stdout.write("\n")
    return 0


def _render_viz(
    args: argparse.Namespace,
    graphs: list[Graph],
    topo: Graph,
    fmt: str,
) -> int:
    from rlmflow.utils.viewer import save_html, save_image, save_steps

    element_mult = args.element_mult
    if element_mult is None:
        element_mult = 2.0 if fmt == "html" else 3.0

    if fmt == "html":
        if not args.out:
            raise SystemExit("rlmflow: --format html requires --out PATH")
        path = save_html(
            graphs,
            args.out,
            title=args.title or "rlmflow run",
            element_mult=element_mult,
            marker_mult=args.marker_mult,
            text_mult=args.text_mult,
            normalize_labels=args.normalize_labels,
        )
        print(f"wrote {path}", file=sys.stderr)
        return 0

    if fmt == "image":
        if not args.out:
            raise SystemExit(
                "rlmflow: --format image requires --out PATH (e.g. graph.png)"
            )
        path = save_image(
            topo,
            args.out,
            width=args.width,
            height=args.height,
            scale=args.scale,
            element_mult=element_mult,
            marker_mult=args.marker_mult,
            text_mult=args.text_mult,
            normalize_labels=args.normalize_labels,
        )
        print(f"wrote {path}", file=sys.stderr)
        return 0

    if fmt == "steps":
        if not args.out:
            raise SystemExit("rlmflow: --format steps requires --out DIR")
        path = save_steps(
            graphs,
            args.out,
            fmt=args.image_format,
            width=args.width,
            height=args.height,
            scale=args.scale,
            element_mult=element_mult,
            marker_mult=args.marker_mult,
            text_mult=args.text_mult,
            normalize_labels=args.normalize_labels,
        )
        print(f"wrote {len(graphs)} images under {path}", file=sys.stderr)
        return 0

    raise SystemExit(f"rlmflow: unknown viz format {fmt!r}")


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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rlmflow",
        description="rlmflow command-line tools",
    )
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

    v = sub.add_parser(
        "view",
        help="open the Gradio viewer on a workspace",
    )
    v.add_argument("path", help="workspace directory")
    v.add_argument("--share", action="store_true", help="create a public URL")
    v.add_argument("--port", type=int, default=None, help="server port")
    v.add_argument("--host", default=None, help="server host / bind address")
    v.set_defaults(func=cmd_view)

    r = sub.add_parser(
        "render",
        help="render a workspace in any of several formats",
    )
    r.add_argument("path", help="workspace directory")
    r.add_argument(
        "--format",
        "-f",
        required=True,
        choices=[
            "mermaid",
            "mermaid-flowchart",
            "mermaid-sequence",
            "dot",
            "d2",
            "tree",
            "ascii-boxes",
            "gantt-html",
            "report-md",
            "code-log",
            "error-summary",
            "tokens",
            "html",
            "image",
            "steps",
        ],
        help="output format",
    )
    r.add_argument(
        "--out",
        "-o",
        default=None,
        help=(
            "write to file (default: stdout). Required for 'html', "
            "'image', and 'steps' formats."
        ),
    )
    r.add_argument("--title", default=None, help="title for --format html")
    r.add_argument(
        "--width", type=int, default=1800, help="image canvas width in pixels"
    )
    r.add_argument(
        "--height", type=int, default=1350, help="image canvas height in pixels"
    )
    r.add_argument(
        "--scale", type=float, default=2.0, help="kaleido density multiplier"
    )
    r.add_argument(
        "--element-mult",
        type=float,
        default=None,
        help="uniform marker + edge + font multiplier",
    )
    r.add_argument(
        "--marker-mult",
        type=float,
        default=None,
        help="marker + edge multiplier (overrides --element-mult)",
    )
    r.add_argument(
        "--text-mult",
        type=float,
        default=None,
        help="font multiplier (overrides --element-mult)",
    )
    r.add_argument(
        "--normalize-labels",
        dest="normalize_labels",
        action="store_true",
        default=True,
        help="force every label to bottom-center (default for image / steps / html)",
    )
    r.add_argument(
        "--no-normalize-labels",
        dest="normalize_labels",
        action="store_false",
        help="keep the alternating top/bottom label layout",
    )
    r.add_argument(
        "--image-format",
        default="png",
        help="image suffix for --format steps (default: png)",
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
