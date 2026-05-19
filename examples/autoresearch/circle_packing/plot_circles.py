"""Plot archived circle-packing candidates.

Examples:
    python plot_circles.py ../runs/autoresearch/history/22_lbfgs_polish.py
    python plot_circles.py ../runs/autoresearch/history/22_lbfgs_polish.py \
        ../runs/autoresearch/history/26_swap_heuristic.py --out top.png
    python plot_circles.py ../runs/autoresearch/history --top 8 --out top8.png
    python plot_circles.py ../runs/autoresearch/history/ledger.jsonl --top 8
    python plot_circles.py ../runs/autoresearch/history/ledger.jsonl \
        --best-by-algorithm --top 12 --out top12_algorithms.png
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


N = 26


@dataclass(frozen=True)
class Candidate:
    """One candidate source file plus optional score metadata from a ledger."""

    path: Path
    label: str
    score: float | None = None


@dataclass(frozen=True)
class Packing:
    """The evaluated geometry returned by one candidate ``solve()`` function."""

    candidate: Candidate
    centers: np.ndarray
    radii: np.ndarray

    @property
    def score(self) -> float:
        """Return the score used by the evaluator: sum of all radii."""

        return float(np.sum(self.radii))


def _numeric_prefix(path: Path) -> tuple[int, str]:
    """Sort archived files by their leading run number, then by name."""

    match = re.match(r"(\d+)_", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def _candidate_label(path: Path) -> str:
    """Convert ``22_lbfgs_polish.py`` into a compact plot label."""

    return path.stem


def _algorithm_name(candidate: Candidate) -> str:
    """Return the strategy-family name for a candidate.

    Examples:
      ``30_multi_start_local`` -> ``multi_start_local``
      ``31_multi_start_local_fix1`` -> ``multi_start_local``
    """

    name = candidate.label or candidate.path.stem
    name = re.sub(r"^\d+_", "", name)
    return re.sub(r"_fix\d+$", "", name)


def _packing_score(packing: Packing) -> float:
    """Use ledger score when present, otherwise recompute from radii."""

    if packing.candidate.score is not None:
        return packing.candidate.score
    return packing.score


def _best_by_algorithm(
    packings: list[Packing], *, include_baseline: bool = False
) -> list[Packing]:
    """Keep only the best scored packing from each strategy family."""

    best: dict[str, Packing] = {}
    for packing in packings:
        algorithm = _algorithm_name(packing.candidate)
        if algorithm == "baseline" and not include_baseline:
            continue
        current = best.get(algorithm)
        if current is None or _packing_score(packing) > _packing_score(current):
            best[algorithm] = packing
    return sorted(best.values(), key=_packing_score, reverse=True)


def _resolve_ledger_path(ledger: Path, solution_path: str) -> Path:
    """Resolve a ledger ``solution_path`` relative to the run root.

    Ledger rows store paths like ``history/22_lbfgs_polish.py`` relative to
    the autoresearch run directory, while the ledger itself lives under
    ``<run>/history/ledger.jsonl``.
    """

    raw = Path(solution_path)
    if raw.is_absolute():
        return raw
    run_root = ledger.parent.parent
    return run_root / raw


def _read_ledger(path: Path) -> list[Candidate]:
    """Read scored candidates from ``ledger.jsonl`` in ledger order."""

    candidates: list[Candidate] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("score") is None:
            continue
        source_path = row.get("solution_path")
        if not source_path:
            continue
        candidates.append(
            Candidate(
                path=_resolve_ledger_path(path, source_path),
                label=f"{row.get('n')}_{row.get('description', 'candidate')}",
                score=float(row["score"]),
            )
        )
    return candidates


def _expand_inputs(inputs: list[str]) -> list[Candidate]:
    """Expand files, directories, globs, and ledgers into candidate files."""

    candidates: list[Candidate] = []
    for item in inputs:
        matches = (
            [Path(p) for p in sorted(glob.glob(item))]
            if any(ch in item for ch in "*?[")
            else []
        )
        paths = matches or [Path(item)]
        for path in paths:
            path = path.expanduser()
            if path.name == "ledger.jsonl":
                candidates.extend(_read_ledger(path))
            elif path.is_dir():
                candidates.extend(
                    Candidate(p, _candidate_label(p))
                    for p in sorted(path.glob("*.py"), key=_numeric_prefix)
                )
            else:
                candidates.append(Candidate(path, _candidate_label(path)))

    seen: set[Path] = set()
    unique: list[Candidate] = []
    for candidate in candidates:
        key = candidate.path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _load_solve(path: Path):
    """Import one source file and return its top-level ``solve`` function."""

    module_name = f"candidate_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    solve = getattr(module, "solve", None)
    if solve is None:
        raise RuntimeError("missing top-level solve()")
    return solve


def _evaluate(candidate: Candidate) -> Packing:
    """Run ``solve()`` and coerce the returned centers/radii arrays."""

    solve = _load_solve(candidate.path)
    centers, radii = solve()
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if centers.shape != (N, 2):
        raise ValueError(f"centers shape {centers.shape} != ({N}, 2)")
    if radii.shape != (N,):
        raise ValueError(f"radii shape {radii.shape} != ({N},)")
    return Packing(candidate=candidate, centers=centers, radii=radii)


def _plot_packing(ax, packing: Packing) -> None:
    """Draw one packing in a unit-square subplot."""

    import matplotlib.patches as patches

    centers = packing.centers
    radii = packing.radii
    score = packing.candidate.score if packing.candidate.score is not None else packing.score

    ax.add_patch(
        patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", linewidth=1.2)
    )
    order = np.argsort(radii)
    colors = np.linspace(0.15, 0.95, len(radii))
    for rank, idx in enumerate(order):
        x, y = centers[idx]
        r = radii[idx]
        circle = patches.Circle(
            (float(x), float(y)),
            float(r),
            facecolor=plt.cm.viridis(colors[rank]),
            edgecolor="black",
            linewidth=0.5,
            alpha=0.75,
        )
        ax.add_patch(circle)
        if r > 0.035:
            ax.text(float(x), float(y), str(idx), ha="center", va="center", fontsize=6)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{packing.candidate.label}\nscore={score:.6f} min={np.min(radii):.4f}",
        fontsize=9,
    )


def plot(packings: list[Packing], out: Path | None = None, cols: int = 3) -> None:
    """Render all evaluated packings as a subplot grid."""

    if not packings:
        raise SystemExit("No valid packings to plot.")

    rows = math.ceil(len(packings) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, packing in zip(axes.ravel(), packings, strict=False):
        ax.axis("on")
        _plot_packing(ax, packing)

    fig.suptitle(f"Circle Packing Candidates ({len(packings)} shown)", fontsize=14)
    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")
    else:
        plt.show()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for plotting candidate solution files."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Candidate .py files, a history directory, glob patterns, or ledger.jsonl.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help=(
            "Plot only the top N scored candidates after evaluating all valid inputs. "
            "With --best-by-algorithm, this is the top N algorithm families."
        ),
    )
    parser.add_argument(
        "--best-by-algorithm",
        action="store_true",
        help="Collapse _fixN attempts and plot the best scored run per algorithm family.",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Include the baseline row when using --best-by-algorithm.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=3,
        help="Number of subplot columns.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Save the plot to this image path instead of opening a window.",
    )
    args = parser.parse_args(argv)

    candidates = _expand_inputs(args.inputs)
    packings: list[Packing] = []
    for candidate in candidates:
        try:
            packings.append(_evaluate(candidate))
        except Exception as exc:  # noqa: BLE001
            print(f"skip {candidate.path}: {exc}", file=sys.stderr)

    if args.best_by_algorithm:
        packings = _best_by_algorithm(packings, include_baseline=args.include_baseline)
    else:
        packings = sorted(packings, key=_packing_score, reverse=True)

    if args.top is not None:
        packings = packings[: args.top]

    plot(packings, out=args.out, cols=max(1, args.cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
