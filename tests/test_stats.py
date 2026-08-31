import numpy as np
import pandas as pd

from fpl_model.features.stats import per90, shrink_to_mean, to_score_0_100
from fpl_model.scoring.engine import horizon_weights


def test_horizon_weights_keep_five_gw_spec():
    hz = [0.28, 0.22, 0.18, 0.16, 0.16]
    w = horizon_weights(hz, 5)
    assert len(w) == 5
    np.testing.assert_allclose(w, hz)
    np.testing.assert_allclose(w.sum(), 1.0)


def test_horizon_weights_stretch_to_ten():
    w = horizon_weights([0.28, 0.22, 0.18, 0.16, 0.16], 10)
    assert len(w) == 10
    np.testing.assert_allclose(w.sum(), 1.0)
    assert w[0] > w[-1]
    assert w[0] < 0.40



def test_per90_and_shrink_towards_prior_with_few_minutes():
    rate = pd.Series(per90(pd.Series([2.0, 0.0]), pd.Series([90.0, 0.0])))
    shrunk = shrink_to_mean(rate, pd.Series([90.0, 0.0]), prior_mean=0.4, prior_minutes=450)
    assert rate.iloc[0] == 2.0
    assert shrunk.iloc[0] < 2.0
    assert shrunk.iloc[1] == 0.4


def test_shrink_caps_prior_weight_at_zero_minutes():
    rate = pd.Series([0.0, 0.0])
    prior = pd.Series([2.0, 2.0])
    shrunk = shrink_to_mean(rate, pd.Series([0.0, 0.0]), prior, 450, max_prior_weight=0.85)
    # 85% prior + 15% observed (0)
    np.testing.assert_allclose(shrunk.iloc[0], 1.7)


def test_blend_playing_time_drops_prior_after_grace_with_zero_starts():
    from fpl_model.features.stats import blend_playing_time

    observed = pd.Series([0.0, 0.0, 1.0])
    minutes = pd.Series([0.0, 0.0, 270.0])
    prior = pd.Series([0.8, 0.8, 0.8])
    starts = pd.Series([0, 0, 3])
    matches = pd.Series([1, 4, 4])
    out = blend_playing_time(observed, minutes, prior, 450, 0.55, starts, matches, grace_matches=3)
    assert out.iloc[0] == 0.55 * 0.8  # still blended (grace not met)
    assert out.iloc[1] == 0.0  # zero starts after grace → trust observed
    assert out.iloc[2] > 0.7


def test_mid_overall_uses_position_weights():
    from fpl_model.config import load_settings, load_weights
    from fpl_model.scoring.engine import score_feature_frame

    settings = load_settings()
    weights = load_weights()
    n = 6
    df = pd.DataFrame(
        {
            "position": ["MID"] * n,
            "team_id": [1] * n,
            "price": [5.0] * n,
            "start_pct": [0.9] * n,
            "minutes_pct": [0.9] * n,
            "start_pct_l5": [0.9] * n,
            "availability": [1.0] * n,
            "matches": [10] * n,
            "pitch_attack": [92, 92, 92, 40, 40, 40],
            "pitch_central": [90] * n,
            "understat_matched": [1] * n,
            "npxgi90": [0.5, 0.5, 0.5, 0.1, 0.1, 0.1],
            "npxgi90_l5": [0.6, 0.5, 0.4, 0.1, 0.1, 0.1],
            "npxg90": [0.3, 0.3, 0.3, 0.05, 0.05, 0.05],
            "npxa90": [0.2] * n,
            "defcon_hit_rate": [0.0, 0.0, 0.0, 0.8, 0.8, 0.8],
            "points90": [5.0] * n,
            "goals90": [0.3] * n,
            "assists90": [0.2] * n,
            "bonus90": [0.2] * n,
            "bps90": [20.0] * n,
            "team_attack": [1.5] * n,
            "team_defence": [-1.2] * n,
            "fixtures_json": ["{}"] * n,
            "minutes_pct_l5": [0.9] * n,
            "role_change_raw": [0.0] * n,
            "price_change": [0.0] * n,
            "team_attack_trend": [0.0] * n,
            "team_defence_trend": [0.0] * n,
        }
    )
    scored = score_feature_frame(df, weights, settings)
    am = scored.loc[scored["pitch_attack"] == 92, "U"].mean()
    dm = scored.loc[scored["pitch_attack"] == 40, "U"].mean()
    # Different profiles should not share the same U after AM vs DM mix
    assert am != dm


def test_mid_overall_moves_with_fixtures():
    import json

    from fpl_model.config import load_settings, load_weights
    from fpl_model.scoring.engine import score_feature_frame

    settings = load_settings()
    weights = load_weights()
    assert weights["overall_by_pos"]["MID"]["O"] > 0
    for pos in ("DEF", "MID", "FWD", "GK"):
        assert weights["overall_by_pos"][pos]["O"] > 0

    def fx(leak: float, att: float, n: int = 5) -> str:
        by = {str(2 + i): [{"home": 1.0, "opp_defence_leak": leak, "opp_attack": att, "p_cs": 0.3}] for i in range(n)}
        return json.dumps(by)

    n = 4
    df = pd.DataFrame(
        {
            "position": ["MID"] * n,
            "team_id": [1] * n,
            "price": [6.0] * n,
            "start_pct": [0.9] * n,
            "minutes_pct": [0.9] * n,
            "start_pct_l5": [0.9] * n,
            "availability": [1.0] * n,
            "matches": [10] * n,
            "pitch_attack": [80] * n,
            "pitch_central": [70] * n,
            "understat_matched": [1] * n,
            "npxgi90": [0.4] * n,
            "npxg90": [0.2] * n,
            "npxa90": [0.2] * n,
            "defcon_hit_rate": [0.2] * n,
            "points90": [4.0] * n,
            "goals90": [0.2] * n,
            "assists90": [0.2] * n,
            "bonus90": [0.2] * n,
            "bps90": [18.0] * n,
            "team_attack": [1.5] * n,
            "team_defence": [-1.2] * n,
            "fixtures_json": [fx(2.0, 0.8), fx(2.0, 0.8), fx(0.6, 2.0), "{}"],
            "minutes_pct_l5": [0.9] * n,
            "role_change_raw": [0.0] * n,
            "price_change": [0.0] * n,
            "team_attack_trend": [0.0] * n,
            "team_defence_trend": [0.0] * n,
        }
    )
    scored = score_feature_frame(df, weights, settings)
    easy = scored.loc[0, "overall"]
    hard = scored.loc[2, "overall"]
    blank = scored.loc[3, "overall"]
    assert easy > hard
    assert easy > blank


def test_mc_is_an_attacking_midfield_role():
    from fpl_model.scoring.role import pitch_attack_score

    assert pitch_attack_score("MC") >= 74
    assert pitch_attack_score("AMC") > pitch_attack_score("MC") > pitch_attack_score("DMC")


def test_gk_minutes_reliability_drops_backup():
    from fpl_model.config import load_settings
    from fpl_model.scoring.engine import _apply_minutes_reliability

    settings = load_settings()
    df = pd.DataFrame(
        {
            "M": [80.0, 80.0],
            "matches": [4, 4],
            "start_pct_l5": [1.0, 0.0],
            "start_pct": [1.0, 0.1],
            "team_id": [1, 1],
            "position": ["GK", "GK"],
        }
    )
    idx = df["position"] == "GK"
    _apply_minutes_reliability(df, idx, "GK", settings)
    assert df.loc[0, "M"] > 70
    assert df.loc[1, "M"] < 20



def test_scores_are_position_relative_percentiles():
    s = to_score_0_100(pd.Series([1.0, 2.0, 3.0]))
    assert list(s.round(0)) == [33.0, 67.0, 100.0]
    inv = to_score_0_100(pd.Series([1.0, 2.0, 3.0]), invert=True)
    assert list(inv.round(0)) == [67.0, 33.0, 0.0]
