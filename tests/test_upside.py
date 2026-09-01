import pandas as pd

from fpl_model.config import load_weights
from fpl_model.features.asof import _xg_trend_by_team
from fpl_model.features.stats import signed_to_score
from fpl_model.scoring.upside import compute_upside, ownership_score


def test_wasted_chances_raise_finishing_upside():
    weights = load_weights()
    waste = pd.DataFrame(
        {
            "position": ["MID", "MID"],
            "overall": [70.0, 70.0],
            "npxg90": [0.5, 0.1],
            "goals90": [0.1, 0.5],
            "npxa90": [0.2, 0.2],
            "assists90": [0.2, 0.2],
            "minutes_pct_l5": [0.6, 0.6],
            "minutes_pct": [0.6, 0.6],
            "role_change_raw": [0.0, 0.0],
            "price_change": [0.0, 0.0],
            "team_attack_trend": [0.0, 0.0],
            "team_defence_trend": [0.0, 0.0],
            "last_selected": [100_000, 100_000],
            "matches": [12, 12],
        }
    )
    out = compute_upside(waste, weights)
    assert out.loc[0, "up_finishing"] > 50
    assert out.loc[1, "up_finishing"] < 50
    assert out.loc[0, "up_finishing"] > out.loc[1, "up_finishing"]


def test_signed_to_score_neutral_at_zero():
    s = signed_to_score(pd.Series([0.0, 0.2, -0.2]), default_scale=0.12)
    assert abs(s.iloc[0] - 50) < 1
    assert s.iloc[1] > 50
    assert s.iloc[2] < 50


def test_ownership_is_rank_lever():
    overall = pd.Series([80.0, 80.0, 50.0])
    selected = pd.Series([100.0, 9_000_000.0, 4_000_000.0])
    s = ownership_score(overall, selected)
    assert s.iloc[0] > 50
    assert s.iloc[1] < 50
    assert abs(s.iloc[2] - 50) < 1


def test_xg_trend_attack_up_defence_better():
    ut = pd.DataFrame(
        {
            "team_id": [1] * 10,
            "date": [f"2025-08-{d:02d}" for d in range(1, 11)],
            "xg_for": [1.0] * 5 + [2.0] * 5,
            "xg_against": [1.5] * 5 + [0.8] * 5,
        }
    )
    t = _xg_trend_by_team(ut)
    assert float(t["team_attack_trend"].iloc[0]) > 0
    assert float(t["team_defence_trend"].iloc[0]) > 0


def test_compute_upside_not_stuck_at_fifty():
    weights = load_weights()
    df = pd.DataFrame(
        {
            "position": ["MID", "MID", "MID", "GK"],
            "overall": [80, 55, 72, 60],
            "npxg90": [0.4, 0.2, 0.3, 0.0],
            "goals90": [0.1, 0.4, 0.3, 0.0],
            "npxa90": [0.3, 0.1, 0.2, 0.0],
            "assists90": [0.1, 0.3, 0.2, 0.0],
            "minutes_pct_l5": [0.9, 0.3, 0.6, 1.0],
            "minutes_pct": [0.5, 0.8, 0.6, 1.0],
            "role_change_raw": [10.0, -8.0, 0.0, 0.0],
            "price_change": [-0.4, 0.5, 0.0, 0.0],
            "team_attack_trend": [0.4, -0.3, 0.0, 0.2],
            "team_defence_trend": [0.1, -0.2, 0.0, 0.3],
            "last_selected": [1_000, 8_000_000, 500_000, 2_000_000],
            "matches": [12, 12, 12, 12],
        }
    )
    out = compute_upside(df, weights)
    assert out["upside"].std() > 3
    assert out.loc[3, "up_finishing"] == 50.0
    assert out.loc[0, "upside"] > out.loc[1, "upside"]
