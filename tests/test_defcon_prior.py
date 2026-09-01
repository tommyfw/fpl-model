import pandas as pd

from fpl_model.features.asof import _blend_defcon_hit_rate


def test_defcon_blend_uses_prior_when_no_starts():
    out = pd.DataFrame(
        {
            "defcon_hit_rate": [0.0, 0.0],
            "position": ["MID", "DEF"],
            "fbref_cbit90": [14.0, None],
        },
        index=[1, 2],
    )
    n = pd.Series({1: 0.0, 2: 0.0})
    season_prior = pd.Series({1: 0.55, 2: 0.40})
    blended = _blend_defcon_hit_rate(
        out,
        n,
        season_prior=season_prior,
        prior_n=6,
        thresholds={"MID": 12, "DEF": 10, "FWD": 12, "GK": None},
    )
    assert abs(float(blended.loc[1]) - 0.55) < 1e-9
    assert abs(float(blended.loc[2]) - 0.40) < 1e-9


def test_defcon_blend_moves_toward_observed():
    out = pd.DataFrame(
        {
            "defcon_hit_rate": [1.0],
            "position": ["MID"],
            "fbref_cbit90": [None],
        },
        index=[1],
    )
    n = pd.Series({1: 6.0})
    season_prior = pd.Series({1: 0.0})
    blended = _blend_defcon_hit_rate(
        out,
        n,
        season_prior=season_prior,
        prior_n=6,
        thresholds={"MID": 12, "DEF": 10, "FWD": 12, "GK": None},
    )
    # n=6, prior_n=6 → 0.5 * 1 + 0.5 * 0
    assert abs(float(blended.loc[1]) - 0.5) < 1e-9
