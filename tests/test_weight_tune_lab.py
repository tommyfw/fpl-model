"""Lab grid helpers (experiments/weight_tune). Isolated from live weights."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

LAB = Path(__file__).resolve().parents[1] / "experiments" / "weight_tune"
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from run import (  # noqa: E402
    apply_parent_mixes,
    iter_candidates,
    mix_relevant_for_pred,
    parse_mix_label,
    ranking_settings,
    utilities_by_mix,
)


def test_parse_mix_label_pads_legacy_two_part():
    assert parse_mix_label("more_m__more_u") == ("more_m", "more_u", "yaml_as_is")
    assert parse_mix_label("a__b__c") == ("a", "b", "c")


def test_iter_candidates_is_factored_not_cartesian():
    base = {
        "overall": {"Q": 0.4, "M": 0.4, "O": 0.14, "V": 0.06},
        "quality": {"U": 0.5, "F": 0.22, "R": 0.18, "T": 0.1},
        "xpts": {"cs_scale": 0.65, "start_l5": 0.55},
    }
    grids = {
        "overall": {"yaml_as_is": None, "more_m": {"boost": "M", "by": 0.08}},
        "quality": {"yaml_as_is": None, "more_u": {"boost": "U", "by": 0.08}},
        "xpts": {"yaml_as_is": None, "cs_down": {"cs_scale": -0.15}},
    }
    cands = iter_candidates(base, grids)
    labels = [lab for lab, _ in cands]
    assert len(cands) == 5
    assert "yaml_as_is__yaml_as_is__yaml_as_is" in labels
    assert "more_m__more_u__yaml_as_is" in labels
    assert "yaml_as_is__yaml_as_is__cs_down" in labels
    assert "more_m__yaml_as_is__cs_down" not in labels
    w = dict(cands)
    assert abs(w["yaml_as_is__yaml_as_is__cs_down"]["xpts"]["cs_scale"] - 0.50) < 1e-9
    assert w["yaml_as_is__yaml_as_is__cs_down"]["xpts"]["start_l5"] == 0.55
    more = w["more_m__yaml_as_is__yaml_as_is"]["overall"]
    assert more["M"] > 0.4
    assert abs(sum(more.values()) - 1.0) < 1e-9
    assert mix_relevant_for_pred("more_m__yaml_as_is__yaml_as_is", "overall")
    assert not mix_relevant_for_pred("more_m__yaml_as_is__yaml_as_is", "xpts")
    assert mix_relevant_for_pred("yaml_as_is__yaml_as_is__cs_down", "xpts")
    assert not mix_relevant_for_pred("yaml_as_is__yaml_as_is__cs_down", "overall")


def test_nudge_uses_live_by_pos_not_one_global_mix():
    from run import nudge_simplex

    base = {
        "overall": {"Q": 0.36, "M": 0.50, "O": 0.10, "V": 0.04},
        "overall_by_pos": {
            "DEF": {"Q": 0.36, "M": 0.46, "O": 0.12, "V": 0.06},
            "MID": {"Q": 0.36, "M": 0.50, "O": 0.08, "V": 0.06},
        },
    }
    w = apply_parent_mixes(base, overall={"boost": "M", "by": 0.08})
    assert w["overall"]["M"] > 0.50
    assert w["overall_by_pos"]["DEF"]["M"] > 0.46
    assert w["overall_by_pos"]["MID"]["M"] > 0.50
    assert w["overall_by_pos"]["DEF"]["M"] != w["overall_by_pos"]["MID"]["M"]
    n = nudge_simplex({"Q": 0.4, "M": 0.4, "O": 0.14, "V": 0.06}, "M", 0.08)
    assert abs(sum(n.values()) - 1.0) < 1e-9
    assert abs(n["M"] - 0.48) < 1e-9


def test_utilities_weights_recent_season():
    rows = []
    for xname, s24, s25 in (("old", 0.9, 0.1), ("new", 0.1, 0.9)):
        for season, sp in (("2024-25", s24), ("2025-26", s25)):
            for pos in ("GK", "DEF", "MID", "FWD"):
                rows.append(
                    {
                        "mix": f"yaml_as_is__yaml_as_is__{xname}",
                        "season": season,
                        "pred": "xpts",
                        "position": pos,
                        "spearman": sp,
                        "top20_overlap": sp,
                    }
                )
    cfg = ranking_settings({"ranking": {"season_weights": {"2024-25": 0.35, "2025-26": 0.65}}})
    util = utilities_by_mix(pd.DataFrame(rows), "xpts", cfg)
    best = util.sort_values("final_u", ascending=False).iloc[0]["mix"]
    assert best == "yaml_as_is__yaml_as_is__new"


def test_apply_xpts_overlay_adds_delta_and_keeps_other_keys():
    base = {"xpts": {"cs_scale": 0.65, "assist": 3.0}}
    w = apply_parent_mixes(base, xpts={"cs_scale": -0.15})
    assert abs(w["xpts"]["cs_scale"] - 0.50) < 1e-9
    assert w["xpts"]["assist"] == 3.0
