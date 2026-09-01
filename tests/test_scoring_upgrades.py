import json

import pandas as pd

from fpl_model.config import load_settings, load_weights
from fpl_model.ingest.understat import shot_xg_by_match
from fpl_model.scoring.engine import score_feature_frame
from fpl_model.scoring.xpts import compute_xpts


def _base(**kwargs):
    n = 2
    d = {
        "position": ["DEF", "DEF"],
        "price": [4.5, 4.5],
        "start_pct": [1.0, 1.0],
        "minutes_pct": [0.95, 0.95],
        "start_pct_l5": [1.0, 1.0],
        "availability": [1.0, 1.0],
        "matches": [10, 10],
        "npxgi90": [0.05, 0.25],
        "npxg90": [0.02, 0.12],
        "npxa90": [0.03, 0.12],
        "defcon_hit_rate": [0.5, 0.2],
        "p_clean_sheet": [0.35, 0.25],
        "team_defence": [-1.0, -1.3],
        "team_attack": [1.2, 1.2],
        "points90": [4.0, 4.0],
        "goals90": [0.05, 0.1],
        "assists90": [0.05, 0.1],
        "bonus90": [0.1, 0.4],
        "bps90": [18.0, 28.0],
        "pitch_attack": [30, 72],
        "pitch_central": [70, 22],
        "def_fb_share": [0.0, 1.0],
        "understat_matched": [1, 1],
        "fixtures_json": ["{}"] * n,
        "minutes_pct_l5": [0.95, 0.95],
        "role_change_raw": [0.0, 0.0],
        "price_change": [0.0, 0.0],
        "team_attack_trend": [0.0, 0.0],
        "team_defence_trend": [0.0, 0.0],
        "rest_factor": [1.0, 1.0],
    }
    d.update(kwargs)
    return pd.DataFrame(d)


def test_weight_blocks_sum_to_one():
    w = load_weights()
    for key, leaf in w["underlying"].items():
        assert abs(sum(leaf.values()) - 1.0) < 1e-6, key


def test_unmatched_understat_pulls_u_toward_50():
    settings = load_settings()
    weights = load_weights()
    df = _base(understat_matched=[1, 0])
    scored = score_feature_frame(df, weights, settings)
    assert abs(float(scored.loc[1, "U"]) - 50.0) < abs(float(scored.loc[0, "U"]) - 50.0)


def test_cb_u_beats_fb_on_cs_profile():
    settings = load_settings()
    weights = load_weights()
    df = _base()
    scored = score_feature_frame(df, weights, settings)
    assert scored.loc[0, "U"] != scored.loc[1, "U"]


def test_gk_u_does_not_use_fpl_output_leaf():
    w = load_weights()
    assert "fpl_output" not in w["underlying"]["GK"]
    assert "shot_stopping" in w["underlying"]["GK"]


def test_opponent_leak_lifts_attack_xpts():
    settings = load_settings()
    weights = load_weights()
    weak = json.dumps({"2": [{"home": 1, "opp_defence_leak": 0.75, "opp_attack": 1.2, "p_cs": 0.2, "own_xg": 1.4}]})
    strong = json.dumps({"2": [{"home": 1, "opp_defence_leak": 1.9, "opp_attack": 1.2, "p_cs": 0.2, "own_xg": 1.4}]})
    cols = {
        "position": ["MID", "MID"],
        "start_pct": [1.0, 1.0],
        "start_pct_l5": [1.0, 1.0],
        "minutes_pct": [0.95, 0.95],
        "availability": [1.0, 1.0],
        "matches": [10, 10],
        "npxg90": [0.3, 0.3],
        "npxa90": [0.2, 0.2],
        "saves90": [0.0, 0.0],
        "points90": [5.0, 5.0],
        "bonus90": [0.2, 0.2],
        "bps90": [20.0, 20.0],
        "defcon_hit_rate": [0.2, 0.2],
        "fixtures_json": [weak, strong],
    }
    x = compute_xpts(pd.DataFrame(cols), weights, settings)
    assert x.iloc[1] > x.iloc[0]


def test_bonus_raises_attacker_xpts():
    settings = load_settings()
    weights = load_weights()
    fx = json.dumps({"2": [{"home": 1, "opp_defence_leak": 1.2, "opp_attack": 1.2, "p_cs": 0.2}]})
    low = pd.DataFrame(
        {
            "position": ["MID", "MID"],
            "start_pct": [1.0, 1.0],
            "start_pct_l5": [1.0, 1.0],
            "minutes_pct": [0.95, 0.95],
            "availability": [1.0, 1.0],
            "matches": [10, 10],
            "npxg90": [0.3, 0.3],
            "npxa90": [0.2, 0.2],
            "bonus90": [0.05, 0.8],
            "bps90": [12.0, 40.0],
            "defcon_hit_rate": [0.1, 0.1],
            "saves90": [0.0, 0.0],
            "fixtures_json": [fx, fx],
        }
    )
    xp = compute_xpts(low, weights, settings)
    assert xp.iloc[1] > xp.iloc[0]


def test_rest_cuts_xpts():
    settings = load_settings()
    weights = load_weights()
    fx = json.dumps({"2": [{"home": 1, "opp_defence_leak": 1.2, "opp_attack": 1.2, "p_cs": 0.3}]})
    df = pd.DataFrame(
        {
            "position": ["FWD", "FWD"],
            "start_pct": [1.0, 1.0],
            "start_pct_l5": [1.0, 1.0],
            "minutes_pct": [0.95, 0.95],
            "availability": [1.0, 1.0],
            "matches": [10, 10],
            "npxg90": [0.5, 0.5],
            "npxa90": [0.1, 0.1],
            "bonus90": [0.2, 0.2],
            "bps90": [20.0, 20.0],
            "defcon_hit_rate": [0.0, 0.0],
            "saves90": [0.0, 0.0],
            "rest_factor": [1.0, 0.82],
            "fixtures_json": [fx, fx],
        }
    )
    xp = compute_xpts(df, weights, settings)
    assert xp.iloc[0] > xp.iloc[1]


def test_shot_xg_splits_skip_penalties():
    shots = [
        {"season": "2026", "match_id": "1", "situation": "OpenPlay", "xG": 0.4},
        {"season": "2026", "match_id": "1", "situation": "Penalty", "xG": 0.76},
        {"season": "2026", "match_id": "1", "situation": "FromCorner", "xG": 0.1},
    ]
    open_xg, sp = shot_xg_by_match(shots, "2026")
    assert abs(open_xg["1"] - 0.4) < 1e-9
    assert abs(sp["1"] - 0.1) < 1e-9
