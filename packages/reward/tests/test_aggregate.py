import math

import numpy as np

from diffusion_omni.reward.aggregate import aggregate, to_markdown


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
    assert kd["n_seeds"] == 3 and kd["fd_n"] == 3
    assert math.isclose(kd["fd"], 3.0)
    assert math.isclose(kd["fd_std"], float(np.std([1, 2, 6], ddof=1)))  # sample std by default
    assert math.isclose(aggregate(_rows(), ddof=0)[1]["fd_std"], float(np.std([1, 2, 6])))
    assert math.isclose(kd["recall"], 0.6)


def test_single_rows_have_no_std():
    agg = {(r["model"], r["nfe"]): r for r in aggregate(_rows())}
    assert agg[("ref", 4)]["fd_std"] is None and agg[("kd", 1)]["fd_std"] is None
    assert math.isnan(agg[("ref", 4)]["ms_per_sample"])


def test_missing_values_are_counted_and_non_numeric_skipped():
    rows = [{"model": "m", "nfe": 1, "seed": s, "fd": v, "tag": "x"} for s, v in [(0, 1.0), (1, float("nan")), (2, 3.0)]]
    rows[0]["recall"] = 0.5
    m = aggregate(rows)[0]
    assert m["n_seeds"] == 3 and m["fd_n"] == 2 and math.isclose(m["fd"], 2.0)
    assert m["recall_std"] is None and m["recall_n"] == 1  # one value: no std
    assert "tag" not in m
    md = to_markdown(aggregate(rows), expected=3)
    assert "(2/3)" in md and "(1/3)" in md


def test_distinct_seeds_counted_once():
    rows = [{"model": "m", "nfe": 1, "seed": 0, "fd": 1.0}, {"model": "m", "nfe": 1, "seed": 0, "fd": 2.0}]
    assert aggregate(rows)[0]["n_seeds"] == 1


def test_markdown():
    md = to_markdown(aggregate(_rows()), expected=3)
    lines = md.splitlines()
    assert lines[0].startswith("| model | nfe | fd |")
    assert "n_seeds" not in lines[0] and "_std" not in lines[0] and "seeded" not in lines[0]
    assert "| ref | 4 | 1 |" in md  # reference row: no std, no k/N tag
    assert "3 ± 2.6 |" in md and "(1/3)" in md  # the seed-0-only nfe=1 row is tagged
