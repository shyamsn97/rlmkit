"""rlmflow hero animation.

Graph-first README animation. This intentionally skips an intro to RLMs:
the video should immediately show what rlmflow offers - a recursive agent
run as a clean, typed execution graph.

Run from the repo root.

Quick preview (480p, fast)::

    manim -pql docs/rlm_animation.py RLMFlowHero

High quality MP4 (1080p, the canonical render)::

    manim -qh docs/rlm_animation.py RLMFlowHero
    cp media/videos/rlm_animation/1080p60/RLMFlowHero.mp4 docs/rlm_animation.mp4
"""

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Circle,
    Create,
    DashedLine,
    FadeIn,
    FadeOut,
    Flash,
    GrowFromCenter,
    Line,
    RegularPolygon,
    RoundedRectangle,
    Scene,
    Text,
    VGroup,
)

BG = "#0B0F14"
WHITE_C = "#E6EDF3"
DIM = "#6E7681"

Q_C = "#F4B860"    # query
A_C = "#5BC0EB"    # action / child call
S_C = "#A685E2"    # supervising / wait / resume
R_C = "#7BD389"    # result
HOT = "#FFD60A"    # focus only

FS_HEADER = 24
FS_BODY = 14
FS_CAP = 12
FS_SMALL = 10
FS_TINY = 8

CODE_FONT = "Menlo"


def typed_node(kind, label="", *, r=0.30, fs=FS_BODY):
    """Small typed graph node."""
    if kind == "Q":
        shape = Circle(radius=r, color=Q_C, fill_opacity=0.18, stroke_width=2)
    elif kind == "A":
        shape = RoundedRectangle(
            corner_radius=0.10,
            width=r * 2.1,
            height=r * 1.55,
            color=A_C,
            fill_opacity=0.18,
            stroke_width=2,
        )
    elif kind == "S":
        shape = RegularPolygon(n=4, color=S_C, fill_opacity=0.18, stroke_width=2)
        shape.rotate(0.7853981633974483).scale(r)
    elif kind == "R":
        outer = Circle(radius=r, color=R_C, fill_opacity=0.0, stroke_width=2)
        inner = Circle(radius=r * 0.72, color=R_C, fill_opacity=0.20, stroke_width=2)
        shape = VGroup(outer, inner)
    else:
        shape = Circle(radius=r, color=DIM, stroke_width=1)

    if not label:
        return VGroup(shape)

    txt = Text(label, font=CODE_FONT, font_size=fs, color=WHITE_C)
    max_w = r * 1.7
    if txt.width > max_w:
        txt.scale(max_w / txt.width)
    return VGroup(shape, txt)


def tiny_graph(specs, *, root_pos, scale=1.0):
    """Small static graph snapshot for the 'evolves every step' phase."""
    nodes = []
    for label, kind, _children in specs:
        nodes.append(typed_node(kind, label, r=0.18 * scale, fs=max(int(7 * scale), 7)))

    def layout(idx, pos, depth):
        nodes[idx].move_to(pos)
        children = specs[idx][2]
        if not children:
            return
        spread = 0.92 * scale / max(depth, 1)
        start = pos[0] - spread * (len(children) - 1) / 2
        for j, child_idx in enumerate(children):
            child_pos = pos + DOWN * 0.62 * scale + RIGHT * (start - pos[0] + spread * j)
            layout(child_idx, child_pos, depth + 1)

    layout(0, root_pos, 1)

    edges = VGroup()
    for idx, (_label, _kind, children) in enumerate(specs):
        for child_idx in children:
            edges.add(Line(
                nodes[idx].get_bottom(),
                nodes[child_idx].get_top(),
                color=DIM,
                stroke_width=1.1,
            ).set_opacity(0.75))
    return VGroup(edges, *nodes)


def fade_clear(scene, *mobjs, run_time=0.55):
    grp = VGroup(*[m for m in mobjs if m is not None])
    if len(grp) > 0:
        scene.play(FadeOut(grp), run_time=run_time)


class RLMFlowHero(Scene):
    """Immediate graph-first animation."""

    def construct(self):
        self.camera.background_color = BG
        self._graph_first_animation()

    def _graph_first_animation(self):
        brand = Text(
            "rlmflow",
            font=CODE_FONT,
            font_size=28,
            color=WHITE_C,
        ).to_corner(UP + LEFT, buff=0.28)

        headline = Text(
            "recursive agents as graphs",
            font=CODE_FONT,
            font_size=FS_HEADER,
            color=WHITE_C,
        ).to_edge(UP, buff=0.55)

        subhead = Text(
            "see every call, wait, resume, and result",
            font=CODE_FONT,
            font_size=FS_CAP,
            color=DIM,
        ).next_to(headline, DOWN, buff=0.22)

        self.play(FadeIn(brand), run_time=0.25)
        self.play(FadeIn(headline, shift=DOWN * 0.08), run_time=0.45)
        self.play(FadeIn(subhead, shift=DOWN * 0.04), run_time=0.35)

        root_q = typed_node("Q", "root", r=0.42, fs=FS_CAP).move_to(UP * 1.70)
        root_a = typed_node("A", "plan", r=0.38, fs=FS_CAP).move_to(UP * 0.85)
        root_s = typed_node("S", "wait", r=0.38, fs=FS_CAP).move_to(UP * 0.02)

        search = typed_node("A", "search", r=0.34, fs=FS_SMALL).move_to(LEFT * 2.45 + DOWN * 0.82)
        code = typed_node("A", "code", r=0.34, fs=FS_SMALL).move_to(DOWN * 0.82)
        verify = typed_node("A", "verify", r=0.34, fs=FS_SMALL).move_to(RIGHT * 2.45 + DOWN * 0.82)

        chunk0 = typed_node("R", "c0", r=0.28, fs=FS_SMALL).move_to(LEFT * 3.20 + DOWN * 1.95)
        chunk1 = typed_node("R", "c1", r=0.28, fs=FS_SMALL).move_to(LEFT * 1.65 + DOWN * 1.95)
        patch = typed_node("R", "patch", r=0.28, fs=FS_TINY).move_to(DOWN * 1.95)
        retry = typed_node("A", "retry", r=0.28, fs=FS_TINY).move_to(RIGHT * 1.65 + DOWN * 1.95)
        ok = typed_node("R", "ok", r=0.28, fs=FS_SMALL).move_to(RIGHT * 3.20 + DOWN * 1.95)

        done = typed_node("R", "done", r=0.42, fs=FS_CAP).move_to(DOWN * 2.95)

        edges = [
            Line(root_q.get_bottom(), root_a.get_top(), color=DIM, stroke_width=1.4),
            Line(root_a.get_bottom(), root_s.get_top(), color=DIM, stroke_width=1.4),
            DashedLine(root_s.get_bottom(), search.get_top(), color=A_C, stroke_width=1.2),
            DashedLine(root_s.get_bottom(), code.get_top(), color=A_C, stroke_width=1.2),
            DashedLine(root_s.get_bottom(), verify.get_top(), color=A_C, stroke_width=1.2),
            Line(search.get_bottom(), chunk0.get_top(), color=DIM, stroke_width=1.0),
            Line(search.get_bottom(), chunk1.get_top(), color=DIM, stroke_width=1.0),
            Line(code.get_bottom(), patch.get_top(), color=DIM, stroke_width=1.0),
            Line(verify.get_bottom(), retry.get_top(), color=HOT, stroke_width=1.0),
            Line(verify.get_bottom(), ok.get_top(), color=DIM, stroke_width=1.0),
            DashedLine(chunk0.get_bottom(), done.get_top(), color=S_C, stroke_width=0.9),
            DashedLine(chunk1.get_bottom(), done.get_top(), color=S_C, stroke_width=0.9),
            DashedLine(patch.get_bottom(), done.get_top(), color=S_C, stroke_width=0.9),
            DashedLine(retry.get_bottom(), done.get_top(), color=S_C, stroke_width=0.9),
            DashedLine(ok.get_bottom(), done.get_top(), color=S_C, stroke_width=0.9),
        ]

        graph_nodes = [
            root_q,
            root_a,
            root_s,
            search,
            code,
            verify,
            chunk0,
            chunk1,
            patch,
            retry,
            ok,
            done,
        ]

        self.play(GrowFromCenter(root_q), run_time=0.55)
        self.play(Create(edges[0]), GrowFromCenter(root_a), run_time=0.55)
        self.play(Create(edges[1]), GrowFromCenter(root_s), run_time=0.55)
        self.play(
            Create(edges[2]),
            Create(edges[3]),
            Create(edges[4]),
            GrowFromCenter(search),
            GrowFromCenter(code),
            GrowFromCenter(verify),
            run_time=0.95,
        )
        self.play(
            Create(edges[5]),
            Create(edges[6]),
            Create(edges[7]),
            Create(edges[8]),
            Create(edges[9]),
            GrowFromCenter(chunk0),
            GrowFromCenter(chunk1),
            GrowFromCenter(patch),
            GrowFromCenter(retry),
            GrowFromCenter(ok),
            run_time=1.05,
        )
        self.play(
            *[Create(edge) for edge in edges[10:]],
            GrowFromCenter(done),
            run_time=1.05,
        )

        footer = Text(
            "deep trees stay readable",
            font=CODE_FONT,
            font_size=FS_BODY,
            color=WHITE_C,
        ).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(footer, shift=UP * 0.06), run_time=0.40)
        self.play(Flash(done, color=R_C, flash_radius=0.55, line_length=0.12), run_time=0.55)
        self.wait(1.45)

        # The older animation's strongest beat: the graph itself evolves
        # step by step, so every intermediate node is a usable handle.
        self.play(
            FadeOut(footer),
            FadeOut(headline),
            FadeOut(subhead),
            *[FadeOut(node) for node in graph_nodes],
            *[FadeOut(edge) for edge in edges],
            run_time=0.55,
        )

        step_title = Text(
            "the graph evolves every step",
            font=CODE_FONT,
            font_size=FS_HEADER,
            color=WHITE_C,
        ).to_edge(UP, buff=0.85)

        self.play(FadeIn(step_title, shift=DOWN * 0.08), run_time=0.45)

        # One large snapshot at a time. This is slower, clearer, and avoids
        # the overlap caused by putting several dense graphs side by side.
        snapshot_steps = [
            (
                [("root", "Q", [])],
                "root query",
            ),
            (
                [("root", "A", [])],
                "agent writes code",
            ),
            (
                [("root", "S", [1, 2, 3]), ("search", "A", []), ("code", "A", []), ("verify", "A", [])],
                "children are delegated",
            ),
            (
                [
                    ("root", "S", [1, 2, 3]),
                    ("search", "A", [4, 5]),
                    ("code", "R", []),
                    ("verify", "A", []),
                    ("c0", "R", []),
                    ("c1", "R", []),
                ],
                "partial results return",
            ),
            (
                [
                    ("root", "S", [1, 2, 3]),
                    ("search", "R", [4, 5]),
                    ("code", "R", []),
                    ("verify", "A", [6]),
                    ("c0", "R", []),
                    ("c1", "R", []),
                    ("retry", "A", []),
                ],
                "one branch retries",
            ),
            (
                [
                    ("root", "R", [1, 2, 3]),
                    ("search", "R", [4, 5]),
                    ("code", "R", []),
                    ("verify", "R", [6, 7]),
                    ("c0", "R", []),
                    ("c1", "R", []),
                    ("retry", "A", []),
                    ("ok", "R", []),
                ],
                "final result",
            ),
        ]

        timeline = Line(LEFT * 4.8 + DOWN * 2.35, RIGHT * 4.8 + DOWN * 2.35, color=DIM, stroke_width=1.2)
        dots = VGroup(*[
            Circle(radius=0.07, color=DIM, fill_opacity=0.35, stroke_width=1.0)
            .move_to(LEFT * 4.8 + RIGHT * (9.6 * i / (len(snapshot_steps) - 1)) + DOWN * 2.35)
            for i in range(len(snapshot_steps))
        ])
        self.play(Create(timeline), FadeIn(dots), run_time=0.55)

        active_graph = None
        active_label = None
        active_dot = None
        for i, (specs, label_text) in enumerate(snapshot_steps):
            graph = tiny_graph(specs, root_pos=UP * 0.95, scale=1.95)
            label = Text(label_text, font=CODE_FONT, font_size=FS_BODY, color=WHITE_C)
            label.move_to(DOWN * 1.88)
            dot = Circle(radius=0.11, color=HOT, fill_opacity=0.65, stroke_width=1.5).move_to(dots[i])
            if active_graph is not None:
                self.play(FadeOut(active_graph), FadeOut(active_label), FadeOut(active_dot), run_time=0.45)
            self.play(
                FadeIn(graph, shift=DOWN * 0.08),
                FadeIn(label, shift=UP * 0.05),
                FadeIn(dot),
                run_time=0.95,
            )
            self.wait(0.75)
            active_graph = graph
            active_label = label
            active_dot = dot

        timeline_footer = Text(
            "checkpoint · fork · inspect at any node",
            font=CODE_FONT,
            font_size=FS_BODY,
            color=WHITE_C,
        ).to_edge(DOWN, buff=0.35)
        self.play(FadeOut(active_label), FadeOut(active_dot), FadeIn(timeline_footer, shift=UP * 0.06), run_time=0.45)
        self.wait(1.55)

        self.play(
            FadeOut(step_title),
            FadeOut(timeline_footer),
            FadeOut(active_graph),
            FadeOut(timeline),
            FadeOut(dots),
            run_time=0.55,
        )

        code_title = Text(
            "agents recursively create child agents in plain Python",
            font=CODE_FONT,
            font_size=FS_BODY,
            color=WHITE_C,
        ).move_to(UP * 2.25)

        code_box = RoundedRectangle(
            corner_radius=0.12,
            width=10.4,
            height=3.55,
            color=DIM,
            stroke_width=1.0,
            fill_opacity=0.06,
        ).move_to(DOWN * 0.12)
        repl_header = Text(
            "REPL",
            font=CODE_FONT,
            font_size=FS_TINY,
            color=DIM,
        ).move_to(code_box.get_corner(UP + RIGHT) + DOWN * 0.18 + LEFT * 0.35)
        ctx_tab = RoundedRectangle(
            corner_radius=0.08,
            width=1.28,
            height=0.36,
            color=Q_C,
            stroke_width=1.2,
            fill_opacity=0.18,
        ).move_to(code_box.get_corner(UP + LEFT) + DOWN * 0.20 + RIGHT * 0.80)
        ctx_label = Text("CONTEXT", font=CODE_FONT, font_size=FS_TINY, color=Q_C).move_to(ctx_tab)
        ctx = VGroup(ctx_tab, ctx_label)
        code_lines = VGroup(
            Text(">>> chunks = CONTEXT.split(by='---')", font=CODE_FONT, font_size=FS_CAP, color=Q_C),
            Text(">>> handles = []", font=CODE_FONT, font_size=FS_CAP, color=WHITE_C),
            Text(">>> for i, chunk in enumerate(chunks):", font=CODE_FONT, font_size=FS_CAP, color=WHITE_C),
            Text("...     h = delegate(f'chunk_{i}', 'analyze', context=chunk)", font=CODE_FONT, font_size=FS_CAP, color=A_C),
            Text("...     handles.append(h)", font=CODE_FONT, font_size=FS_CAP, color=A_C),
            Text(">>> results = yield wait(*handles)", font=CODE_FONT, font_size=FS_CAP, color=S_C),
            Text(">>> done(combine(results))", font=CODE_FONT, font_size=FS_CAP, color=R_C),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.13)
        code_lines.move_to(code_box.get_center() + DOWN * 0.12)

        code_caption = Text(
            "every delegate, wait, and result becomes an inspectable graph",
            font=CODE_FONT,
            font_size=FS_CAP,
            color=DIM,
        ).next_to(code_box, DOWN, buff=0.30)

        self.play(FadeIn(code_title, shift=DOWN * 0.06), run_time=0.65)
        self.play(FadeIn(code_box), FadeIn(repl_header), FadeIn(ctx), run_time=0.70)
        for line in code_lines:
            self.play(FadeIn(line, shift=UP * 0.04), run_time=0.55)
            self.wait(0.10)
        self.play(FadeIn(code_caption, shift=UP * 0.04), run_time=0.55)
        self.wait(2.40)

        fade_clear(
            self,
            brand,
            code_title,
            code_box,
            repl_header,
            ctx,
            code_lines,
            code_caption,
        )
