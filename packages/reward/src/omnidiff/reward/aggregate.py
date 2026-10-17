"""Aggregate scorecard rows over seeds: mean and std per (model, nfe, metric).

Rows are the dicts ``run_scorecard`` returns. ``seed`` is optional. A row that is alone in its (model, nfe) group
(a deterministic reference row, or a single seed) keeps its value and gets no std.
"""

from __future__ import annotations

import math

import numpy as np

KEYS = ("model", "nfe", "seed")


def aggregate(rows: list[dict], ddof: int = 0) -> list[dict]:
    """Group by (model, nfe), in first-seen order. Each metric ``m`` becomes ``m`` (mean) and ``m_std``.

    ``n_seeds`` counts the rows in the group. ``m_std`` is None when the group has one row. NaN values are ignored,
    and a metric that is NaN everywhere stays NaN. ``ddof=0`` matches ``scripts/g2_kl_sweep.py``.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["nfe"]), []).append(r)
    out = []
    for (model, nfe), rs in groups.items():
        row = {"model": model, "nfe": nfe, "n_seeds": len(rs)}
        for c in dict.fromkeys(k for r in rs for k in r if k not in KEYS):
            v = np.array([r[c] for r in rs if r.get(c) is not None and not math.isnan(r[c])], dtype=float)
            row[c] = float(v.mean()) if len(v) else float("nan")
            row[c + "_std"] = float(v.std(ddof=ddof)) if len(rs) > 1 and len(v) > ddof else None
        out.append(row)
    return out


def to_markdown(agg: list[dict]) -> str:
    """Table of ``aggregate`` output: ``mean ± std`` where a std exists, the bare mean otherwise."""
    cols = [c for c in dict.fromkeys(k for r in agg for k in r) if c != "n_seeds" and not c.endswith("_std")]

    def fmt(r, c):
        v = r.get(c)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        if not isinstance(v, float):
            return str(v)
        s = r.get(c + "_std")
        return f"{v:.4g}" if s is None else f"{v:.4g} ± {s:.2g}"

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r, c) for c in cols) + " |" for r in agg]
    return "\n".join(lines)
