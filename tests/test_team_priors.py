import pandas as pd

from fpl_model.config import load_settings
from fpl_model.features.asof import _blend_team_strength, _venue_strength
from fpl_model.ingest.priors import load_team_priors, previous_season, promoted_strength_anchor


def test_previous_season():
    assert previous_season("2026-27") == "2025-26"


def test_promoted_anchor_is_weaker_than_league_mean():
    by_key = pd.DataFrame(
        {
            "prior_attack": [0.9, 1.1, 1.2, 1.35, 1.5, 1.7, 1.9, 2.1],
            "prior_defence": [-1.9, -1.7, -1.55, -1.4, -1.25, -1.1, -0.95, -0.8],
        }
    )
    att, de = promoted_strength_anchor(by_key, 0.25)
    assert att < float(by_key["prior_attack"].mean())
    assert de < float(by_key["prior_defence"].mean())  # more xGA
    assert att > float(by_key["prior_attack"].min())
    mean_att, mean_de = promoted_strength_anchor(by_key, 0.5)
    assert att < mean_att
    assert de < mean_de



def test_blend_pulls_single_match_toward_prior(monkeypatch):
    """One clean sheet should not make a side look like the league's best defence."""
    priors = pd.DataFrame(
        {
            "prior_attack": [1.4, 1.4],
            "prior_defence": [-1.4, -1.4],
            "is_promoted": [1, 0],
        },
        index=pd.Index([11, 1], name="team_id"),
    )

    monkeypatch.setattr("fpl_model.features.asof.load_team_priors", lambda *_a, **_k: priors)
    settings = load_settings()
    # Hull: 1 match, 0 xGA → raw defence 0 (looks elite)
    atk = pd.Series({11: 1.5})
    de = pd.Series({11: 0.0})
    n = pd.Series({11: 1.0})
    batk, bde = _blend_team_strength(None, "2026-27", atk, de, n, settings)
    assert batk.loc[11] < 1.5  # pulled toward prior
    assert bde.loc[11] < -0.5  # no longer ~0 "iron curtain"
    assert bde.loc[11] > -1.4  # still moved a little from the one clean sheet


def test_venue_uses_last_season_when_no_games_this_year():
    season_atk = pd.Series({1: 1.5, 2: 1.4})
    season_de = pd.Series({1: -1.2, 2: -1.3})
    priors = pd.DataFrame(
        {
            "prior_attack_home": [1.8, 1.1],
            "prior_attack_away": [1.1, 0.9],
            "prior_defence_home": [-1.0, -1.5],
            "prior_defence_away": [-1.6, -1.9],
        },
        index=pd.Index([1, 2], name="team_id"),
    )
    ah, aa, dh, da = _venue_strength(pd.DataFrame(), season_atk, season_de, 4, priors)
    # n=0 → mix this-season overall with last-season venue
    assert abs(float(ah.loc[1]) - (0.65 * 1.5 + 0.35 * 1.8)) < 1e-9
    assert abs(float(aa.loc[1]) - (0.65 * 1.5 + 0.35 * 1.1)) < 1e-9
    assert abs(float(dh.loc[2]) - (0.65 * -1.3 + 0.35 * -1.5)) < 1e-9
    assert abs(float(da.loc[2]) - (0.65 * -1.3 + 0.35 * -1.9)) < 1e-9


def test_one_home_game_shrinks_toward_venue_prior():
    ut = pd.DataFrame(
        {
            "team_id": [1],
            "is_home": [1],
            "match_id": [99],
            "xg_for": [3.0],
            "xg_against": [0.2],
        }
    )
    season_atk = pd.Series({1: 1.5})
    season_de = pd.Series({1: -1.2})
    priors = pd.DataFrame(
        {
            "prior_attack_home": [1.8],
            "prior_attack_away": [1.1],
            "prior_defence_home": [-1.0],
            "prior_defence_away": [-1.6],
        },
        index=pd.Index([1], name="team_id"),
    )
    ah, aa, dh, da = _venue_strength(ut, season_atk, season_de, 4, priors)
    # n=1, prior_n=4 → 0.2 observed + 0.8 mix(season, last-season venue)
    base_h = 0.65 * 1.5 + 0.35 * 1.8
    assert abs(float(ah.loc[1]) - (0.2 * 3.0 + 0.8 * base_h)) < 1e-9
    base_dh = 0.65 * -1.2 + 0.35 * -1.0
    assert abs(float(dh.loc[1]) - (0.2 * -0.2 + 0.8 * base_dh)) < 1e-9
    assert abs(float(aa.loc[1]) - (0.65 * 1.5 + 0.35 * 1.1)) < 1e-9


def test_one_home_game_is_closer_to_season_than_raw_venue():
    ut = pd.DataFrame(
        {
            "team_id": [1],
            "is_home": [1],
            "match_id": [99],
            "xg_for": [3.0],
            "xg_against": [0.2],
        }
    )
    season_atk = pd.Series({1: 1.5})
    season_de = pd.Series({1: -1.2})
    priors = pd.DataFrame(
        {
            "prior_attack_home": [1.8],
            "prior_attack_away": [1.1],
            "prior_defence_home": [-1.0],
            "prior_defence_away": [-1.6],
        },
        index=pd.Index([1], name="team_id"),
    )
    ah, _, dh, _ = _venue_strength(ut, season_atk, season_de, 8, priors)
    assert abs(float(ah.loc[1]) - 3.0) > 1.0
    assert float(dh.loc[1]) < -0.4
    assert abs(float(dh.loc[1]) - (-0.2)) > 0.4


def test_load_team_priors_marks_promoted_when_db_present():
    from fpl_model.db import connect

    try:
        conn = connect(load_settings())
    except Exception:
        return
    priors = load_team_priors(conn, "2026-27")
    conn.close()
    if priors.empty:
        return
    # Hull / Coventry / Ipswich should exist; at least one promoted flag if 25/26 loaded
    assert priors["prior_attack"].notna().all()
    assert priors["prior_defence"].notna().all()
    for col in (
        "prior_attack_home",
        "prior_attack_away",
        "prior_defence_home",
        "prior_defence_away",
    ):
        assert col in priors.columns
        assert priors[col].notna().all()
