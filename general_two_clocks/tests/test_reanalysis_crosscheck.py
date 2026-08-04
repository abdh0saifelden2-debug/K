"""Test the committed R1-vs-R2 Helmholtz ratio cross-check (P0-R7)."""
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(HERE, "figures", "27b_reanalysis_crosscheck.json")


def test_crosscheck_artifact():
    with open(ART) as fh:
        d = json.load(fh)
    assert d["minimum_at_500_in_both"] is True
    assert d["max_rel_diff"] < 0.30
    for lev, row in d["levels_hPa"].items():
        assert 0.001 < row["R1"] < 0.10
        assert 0.001 < row["R2"] < 0.10
    # ordering identical: 500 < 250 < 850 in both
    r1 = {k: v["R1"] for k, v in d["levels_hPa"].items()}
    r2 = {k: v["R2"] for k, v in d["levels_hPa"].items()}
    assert sorted(r1, key=r1.get) == sorted(r2, key=r2.get) == ["500", "250", "850"]
