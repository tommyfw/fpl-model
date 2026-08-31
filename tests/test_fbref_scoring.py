import pandas as pd

from fpl_model.config import Settings, load_settings, load_weights
from fpl_model.scoring.engine import score_feature_frame


def _minimal_feats(with_fbref: bool) -> pd.DataFrame:
    n = 6
    base = {
        "fpl_id": list(range(n)),
        "web_name": [f"P{i}" for i in range(n)],
        "position": ["MID"] * n,
        "team_id": [1] * n,
        "npxgi90": [0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
        "npxg90": [0.3, 0.25, 0.2, 0.15, 0.1, 0.05],
        "npxa90": [0.2, 0.15, 0.1, 0.1, 0.05, 0.02],
        "shots_in_box90": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
        "key_passes90": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
        "defcon_hit_rate": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        "xgchain90": [0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
        "chance_quality90": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        "p_clean_sheet": [0.3] * n,
        "team_defence": [-1.0] * n,
        "team_attack": [1.2] * n,
        "saves90": [0.0] * n,
        "points90": [5, 4, 3, 2, 1, 0.5],
        "points90_blend": [5, 4, 3, 2, 1, 0.5],
        "goals90": [0.3, 0.2, 0.1, 0.1, 0.0, 0.0],
        "assists90": [0.2, 0.2, 0.1, 0.1, 0.0, 0.0],
        "bonus90": [0.2] * n,
        "bps90": [20] * n,
        "start_pct": [0.9] * n,
        "minutes_pct": [0.85] * n,
        "start_pct_l5": [0.9] * n,
        "minutes_pct_l5": [0.85] * n,
        "availability": [1.0] * n,
        "pitch_attack": [80] * n,
        "pitch_central": [70] * n,
        "threat90": [40] * n,
        "pen_score": [40] * n,
        "sp_score": [40] * n,
        "price": [8.0] * n,
        "points_form": [5, 4, 3, 2, 1, 0.5],
        "last_selected": [100_000] * n,
        "fixtures_json": ["{}"] * n,
        "price_change": [0.0] * n,
        "role_change_raw": [0.0] * n,
        "team_attack_trend": [0.0] * n,
        "team_defence_trend": [0.0] * n,
        "status": ["a"] * n,
    }
    if with_fbref:
        base["fbref_shots90"] = [4.0, 3.0, 2.0, 1.0, 0.5, 0.2]
        base["fbref_cbit90"] = [8.0, 6.0, 4.0, 2.0, 1.0, 0.5]
        base["fbref_matched"] = [1] * n
    else:
        base["fbref_shots90"] = [float("nan")] * n
        base["fbref_cbit90"] = [float("nan")] * n
        base["fbref_matched"] = [0] * n
    return pd.DataFrame(base)


def test_fbref_leaves_change_underlying():
    settings = load_settings()
    weights = load_weights()
    assert "fbref_shots90" in weights["underlying"]["MID"]
    assert abs(sum(weights["underlying"]["MID"].values()) - 1.0) < 1e-9
    assert weights["underlying"]["MID"]["npxa90"] > weights["underlying"]["MID"]["defcon_hit_rate"]
    no_fb = score_feature_frame(_minimal_feats(False), weights, settings)
    with_fb = score_feature_frame(_minimal_feats(True), weights, settings)
    # Top shot volume player should gain U when FBref is present
    assert with_fb.loc[0, "U"] != no_fb.loc[0, "U"]
    assert with_fb.loc[0, "U"] > with_fb.loc[5, "U"]


def test_missing_fbref_does_not_force_neutral_leaf():
    settings = load_settings()
    weights = load_weights()
    df = _minimal_feats(False)
    scored = score_feature_frame(df, weights, settings)
    assert scored["U"].notna().all()
    assert scored["U"].between(0, 100).all()


def test_team_tilt_moves_team_score():
    settings = load_settings()
    weights = load_weights()
    df = _minimal_feats(False)
    df["team_tilt"] = [0.70, 0.60, 0.50, 0.45, 0.40, 0.30]
    scored = score_feature_frame(df, weights, settings)
    assert scored.loc[0, "T"] > scored.loc[5, "T"]
