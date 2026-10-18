"""Aggregate scorecard rows over seeds: mean and std per (model, nfe, metric).

Rows are the dicts ``run_scorecard`` returns. ``seed`` is optional. A row that is alone in its (model, nfe) group
(a deterministic reference row, or a single seed) keeps its value and gets no std.
"""

from __future__ import annotations

import math

import numpy as np

KEYS = ("model", "nfe", "seed")


def _num(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) and not math.isnan(v)


def aggregate(rows: list[dict], ddof: int = 1) -> list[dict]:
    """Group by (model, nfe), in first-seen order. Each numeric metric ``m`` becomes ``m`` (mean), ``m_std``, ``m_n``.

    ``m_n`` is the number of values that contributed, so a seed whose metric is missing or NaN stays visible.
    ``m_std`` is the sample std (``ddof=1``, n - 1) and is None unless at least two values contributed. A metric
    with no numeric value is NaN. ``n_seeds`` counts distinct seeds in the group (rows without a ``seed`` count
    one each) and ``seeded`` says whether the rows carry seeds. Non-numeric fields are skipped.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["nfe"]), []).append(r)
    out = []
    for (model, nfe), rs in groups.items():
        seeds = {r["seed"] if "seed" in r else i for i, r in enumerate(rs)}
        row = {"model": model, "nfe": nfe, "n_seeds": len(seeds), "seeded": any("seed" in r for r in rs)}
        for c in dict.fromkeys(k for r in rs for k in r if k not in KEYS):
            vals = [r.get(c) for r in rs]
            if any(isinstance(v, (str, list, dict)) for v in vals):
                continue
            v = np.array([x for x in vals if _num(x)], dtype=float)
            row[c] = float(v.mean()) if len(v) else float("nan")
            row[c + "_std"] = float(v.std(ddof=ddof)) if len(v) >= 2 and len(v) > ddof else None
            row[c + "_n"] = len(v)
        out.append(row)
    return out


def to_markdown(agg: list[dict], expected: int | None = None) -> str:
    """Table of ``aggregate`` output: ``mean ± std`` where a std exists, the bare mean otherwise.

    With ``expected`` set, a seeded cell fed by fewer values than that reads ``mean ± std (k/expected)``.
    """
    hidden = lambda c: c in ("n_seeds", "seeded") or c.endswith(("_std", "_n"))
    cols = [c for c in dict.fromkeys(k for r in agg for k in r) if not hidden(c)]

    def fmt(r, c):
        v = r.get(c)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        if not isinstance(v, float):
            return str(v)
        s = r.get(c + "_std")
        text = f"{v:.4g}" if s is None else f"{v:.4g} ± {s:.2g}"
        n = r.get(c + "_n", 1)
        short = expected is not None and r.get("seeded") and n < expected
        return f"{text} ({n}/{expected})" if short else text

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r, c) for c in cols) + " |" for r in agg]
    return "\n".join(lines)
