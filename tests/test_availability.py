import json

import pandas as pd

from fpl_model.config import load_settings, load_weights
from fpl_model.scoring.availability import availability_from_news
from fpl_model.scoring.engine import score_feature_frame


def test_news_out_is_almost_unavailable():
    assert availability_from_news("a", news="Will not play against Arsenal") <= 0.12
    assert availability_from_news("i", chance_this=0, news="") <= 0.12
    assert availability_from_news("d", chance_this=25, news="Doubtful") == 0.25
    assert availability_from_news("a") == 1.0
    assert availability_from_news("a", chance_this=75) == 0.75


def test_injured_starter_minutes_are_capped():
    settings = load_settings()
    weights = load_weights()
    n = 2
    df = pd.DataFrame(
        {
            "position": ["MID"] * n,
            "team_id": [1] * n,
            "price": [8.0] * n,
            "start_pct": [1.0] * n,
            "minutes_pct": [0.95] * n,
            "start_pct_l5": [1.0] * n,
            "availability": [1.0, 0.08],
            "matches": [10] * n,
            "pitch_attack": [80] * n,
            "pitch_central": [70] * n,
            "npxgi90": [0.4] * n,
            "npxg90": [0.2] * n,
            "npxa90": [0.2] * n,
            "defcon_hit_rate": [0.2] * n,
            "points90": [5.0] * n,
            "goals90": [0.2] * n,
            "assists90": [0.2] * n,
            "bonus90": [0.2] * n,
            "bps90": [20.0] * n,
            "team_attack": [1.5] * n,
            "team_defence": [-1.2] * n,
            "fixtures_json": ["{}"] * n,
            "minutes_pct_l5": [0.95] * n,
            "role_change_raw": [0.0] * n,
            "price_change": [0.0] * n,
            "team_attack_trend": [0.0] * n,
            "team_defence_trend": [0.0] * n,
        }
    )
    scored = score_feature_frame(df, weights, settings)
    assert scored.loc[1, "M"] < scored.loc[0, "M"] - 20
    assert scored.loc[1, "M"] < 40


def test_fixture_o_uses_team_xg_context():
    settings = load_settings()
    weights = load_weights()

    def fx(ctx: float) -> str:
        m = {
            "home": 1.0,
            "opp_defence_leak": 1.2,
            "opp_attack": 1.2,
            "p_cs": 0.25,
            "team_xg_context": ctx,
        }
        return json.dumps({str(2 + i): [m] for i in range(5)})

    df = pd.DataFrame(
        {
            "position": ["MID", "MID"],
            "team_id": [1, 1],
            "price": [6.0, 6.0],
            "start_pct": [0.9, 0.9],
            "minutes_pct": [0.9, 0.9],
            "start_pct_l5": [0.9, 0.9],
            "availability": [1.0, 1.0],
            "matches": [10, 10],
            "pitch_attack": [80, 80],
            "pitch_central": [70, 70],
            "npxgi90": [0.4, 0.4],
            "npxg90": [0.2, 0.2],
            "npxa90": [0.2, 0.2],
            "defcon_hit_rate": [0.2] * 2,
            "points90": [4.0, 4.0],
            "goals90": [0.2, 0.2],
            "assists90": [0.2, 0.2],
            "bonus90": [0.2, 0.2],
            "bps90": [18.0, 18.0],
            "team_attack": [1.5, 1.5],
            "team_defence": [-1.2, -1.2],
            "fixtures_json": [fx(80.0), fx(20.0)],
            "minutes_pct_l5": [0.9, 0.9],
            "role_change_raw": [0.0, 0.0],
            "price_change": [0.0, 0.0],
            "team_attack_trend": [0.0, 0.0],
            "team_defence_trend": [0.0, 0.0],
        }
    )
    scored = score_feature_frame(df, weights, settings)
    assert scored.loc[0, "O"] > scored.loc[1, "O"]
