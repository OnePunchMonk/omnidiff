import math

import numpy as np

from omnidiff.reward.aggregate import aggregate, to_markdown


def _rows():
    return [
        {"model": "ref", "nfe": 4, "fd": 1.0, "ms_per_sample": float("nan")},
        {"model": "kd", "nfe": 4, "seed": 0, "fd": 1.0, "recall": 0.5},
        {"model": "kd", "nfe": 4, "seed": 1, "fd": 2.0, "recall": 0.7},
        {"model": "kd", "nfe": 4, "seed": 2, "fd": 6.0, "recall": 0.6},
        {"model": "kd", "nfe": 1, "seed": 0, "fd": 3.0, "recall": 0.1},
    ]


def test_mean_std_and_grouping():
    agg = {(r["model"], r["nfe"]): r for r in aggregate(_rows())}
    assert len(agg) == 3
    kd = agg[("kd", 4)]
    assert kd["n_seeds"] == 3
    assert math.isclose(kd["fd"], 3.0)
    assert math.isclose(kd["fd_std"], float(np.std([1, 2, 6])))
    assert math.isclose(aggregate(_rows(), ddof=1)[1]["fd_std"], float(np.std([1, 2, 6], ddof=1)))
    assert math.isclose(kd["recall"], 0.6)


def test_single_rows_have_no_std():
    agg = {(r["model"], r["nfe"]): r for r in aggregate(_rows())}
    assert agg[("ref", 4)]["fd_std"] is None and agg[("kd", 1)]["fd_std"] is None
    assert math.isnan(agg[("ref", 4)]["ms_per_sample"])


def test_markdown():
    md = to_markdown(aggregate(_rows()))
    lines = md.splitlines()
    assert lines[0].startswith("| model | nfe | fd |")
    assert "n_seeds" not in lines[0] and "_std" not in lines[0]
    assert "| ref | 4 | 1 |" in md
    assert "3 ± 2.2" in md
