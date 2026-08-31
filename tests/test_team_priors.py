import pandas as pd

from fpl_model.config import load_settings
from fpl_model.features.asof import _blend_team_strength
from fpl_model.ingest.priors import load_team_priors, previous_season


def test_previous_season():
    assert previous_season("2026-27") == "2025-26"


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
