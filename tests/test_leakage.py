import pandas as pd

from fpl_model.features.asof import XP_FORBIDDEN, asof_gw_frame, upcoming_fixtures


def test_asof_excludes_current_and_future_rounds():
    gw = pd.DataFrame(
        {
            "fpl_id": [1, 1, 1, 2],
            "round": [8, 9, 10, 10],
            "minutes": [90, 90, 90, 90],
            "xP": [4.1, 5.0, 9.9, 8.0],
        }
    )
    past = asof_gw_frame(gw, as_of_gw=10)
    assert past["round"].max() == 9
    assert (past["round"] < 10).all()
    assert XP_FORBIDDEN not in past.columns


def test_upcoming_fixtures_are_horizon_only():
    fx = pd.DataFrame(
        {
            "event": [9, 10, 11, 14, 15],
            "fixture_id": [1, 2, 3, 4, 5],
            "team_h_score": [1, 2, 0, 3, 1],
        }
    )
    upc = upcoming_fixtures(fx, as_of_gw=10, horizon=5)
    assert set(upc["event"]) == {10, 11, 14}
    assert upc["event"].max() <= 14
    upc10 = upcoming_fixtures(fx, as_of_gw=10, horizon=10)
    assert set(upc10["event"]) == {10, 11, 14, 15}
