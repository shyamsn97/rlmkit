"""Evaluator for circle-packing solutions.

Imports the candidate ``solution.py``, calls ``solve()``, validates
the returned ``(centers, radii)``, and prints ``score: <sum_radii>``
to stdout on success.

Exit codes:

* ``0`` — valid packing; the last stdout line is
  ``score: <float>``.
* ``1`` — any failure (import error, exception in ``solve()``,
  wrong return shape, negative radius, out-of-square, overlap).
  The reason goes to stderr as ``INVALID: <why>`` (plus a Python
  traceback when applicable). **No** ``score:`` line is printed.
  The driver surfaces this as ``ExperimentCrashed`` to the agent.

The agent NEVER sees this file. ``FN_NAME`` below is the source of
truth for the expected function name; the driver reads it via
``_read_evaluator_fn_name`` for preflight checks.

Usage:
    python -u evaluate.py path/to/solution.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
import traceback
from pathlib import Path

import numpy as np


N = 26
FN_NAME = "solve"


def _fail(reason: str) -> int:
    print(f"INVALID: {reason}", file=sys.stderr, flush=True)
    return 1


def _coerce(value):
    """Coerce ``solve()``'s return value into (centers, radii) ndarrays."""
    if not (isinstance(value, tuple) and len(value) == 2):
        return None, (
            f"`{FN_NAME}()` must return a 2-tuple `(centers, radii)`; "
            f"got {type(value).__name__}"
        )
    centers, radii = value
    try:
        centers = np.asarray(centers, dtype=float)
        radii = np.asarray(radii, dtype=float)
    except Exception as e:  # noqa: BLE001
        return None, f"could not coerce return values to float arrays: {e}"
    return (centers, radii), None


def _validate(centers: np.ndarray, radii: np.ndarray) -> str | None:
    if centers.shape != (N, 2):
        return (
            f"centers shape {centers.shape} != ({N}, 2). "
            f"Return exactly {N} centers, each (x, y)."
        )
    if radii.shape != (N,):
        return (
            f"radii shape {radii.shape} != ({N},). "
            f"Return exactly {N} radii."
        )
    eps = 1e-9
    for i in range(N):
        r = float(radii[i])
        x, y = float(centers[i, 0]), float(centers[i, 1])
        if r < 0:
            return f"circle {i} has negative radius {r}"
        if (x - r < -eps or x + r > 1 + eps
                or y - r < -eps or y + r > 1 + eps):
            return (
                f"circle {i} outside unit square: "
                f"center=({x:.4f}, {y:.4f}) r={r:.4f}"
            )
    for i in range(N):
        xi, yi, ri = float(centers[i, 0]), float(centers[i, 1]), float(radii[i])
        for j in range(i + 1, N):
            xj, yj, rj = (
                float(centers[j, 0]),
                float(centers[j, 1]),
                float(radii[j]),
            )
            d = math.hypot(xi - xj, yi - yj)
            if d + eps < ri + rj:
                return (
                    f"circles {i},{j} overlap "
                    f"(d={d:.6f}, r_i+r_j={ri + rj:.6f})"
                )
    return None


def main(path: Path) -> int:
    spec = importlib.util.spec_from_file_location("candidate", str(path))
    if spec is None or spec.loader is None:
        return _fail(f"could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        return _fail(f"import of {path.name} raised; see traceback above")

    fn = getattr(mod, FN_NAME, None)
    if fn is None:
        return _fail(
            f"{path.name} defines no top-level `{FN_NAME}()` function. "
            f"The function name is FIXED — the evaluator does "
            f"`getattr(mod, {FN_NAME!r})()`. Rename your function to "
            f"`{FN_NAME}` and don't add parameters."
        )

    try:
        result = fn()
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        return _fail(f"`{FN_NAME}()` raised; see traceback above")

    coerced, err = _coerce(result)
    if err is not None or coerced is None:
        return _fail(err or "no coerced result")
    centers, radii = coerced

    err = _validate(centers, radii)
    if err is not None:
        return _fail(err)

    sum_r = float(np.sum(radii))
    min_r = float(np.min(radii))
    max_r = float(np.max(radii))
    print(
        f"n={N}  sum_radii={sum_r:.6f}  "
        f"min_r={min_r:.6f}  max_r={max_r:.6f}",
        flush=True,
    )
    print(f"score: {sum_r:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: evaluate.py <solution.py>")
    raise SystemExit(main(Path(sys.argv[1]).resolve()))
