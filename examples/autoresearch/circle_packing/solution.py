"""Baseline solution for circle-packing autoresearch.

The agent rewrites this whole file. The contract is fixed:

* Define a single top-level function ``solve()`` that takes **no
  arguments** and returns ``(centers, radii)``, where ``centers``
  is shape ``(26, 2)`` and ``radii`` is shape ``(26,)``.
* All circles must lie inside ``[0, 1]^2`` and be mutually
  non-overlapping. Score = ``sum(radii)`` (higher is better).
* The function name is FIXED — the evaluator does
  ``getattr(mod, 'solve')()``. Don't rename it, don't add params.
* All imports and helpers must live INSIDE ``solve()`` (or in
  helpers it calls). No module-level state, no
  ``if __name__ == "__main__":`` block.
* Allowed deps: ``numpy`` + Python stdlib only. No scipy / cvxpy
  / shapely / etc.

Strategy here: greedy concentric rings + per-pair radius scaling.
Score ~1.5; known optimum for n=26 is ~2.635.
"""


def solve():
    import numpy as np

    N = 26
    centers = np.zeros((N, 2))
    centers[0] = [0.5, 0.5]
    for i in range(8):
        a = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + 0.3 * np.cos(a), 0.5 + 0.3 * np.sin(a)]
    for i in range(16):
        a = 2 * np.pi * i / 16
        centers[i + 9] = [0.5 + 0.4 * np.cos(a), 0.5 + 0.4 * np.sin(a)]
    centers = np.clip(centers, 0.01, 0.99)

    radii = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )
    for i in range(N):
        for j in range(i + 1, N):
            d = float(np.linalg.norm(centers[i] - centers[j]))
            s = radii[i] + radii[j]
            if s > d:
                scale = d / s
                radii[i] *= scale
                radii[j] *= scale
    return centers, radii
