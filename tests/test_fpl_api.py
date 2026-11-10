import pandas as pd

from fpl_model.ingest.fpl_api import gw_rows_from_live, next_as_of_gw, season_from_bootstrap
from fpl_model.ingest.priors import previous_season


def test_previous_season():
    assert previous_season("2026-27") == "2025-26"
    assert previous_season("2025-26") == "2024-25"


def test_season_from_bootstrap_august():
    boot = {"events": [{"id": 1, "deadline_time": "2026-08-14T17:30:00Z", "finished": False, "is_next": True}]}
    assert season_from_bootstrap(boot) == "2026-27"
    assert next_as_of_gw(boot) == 1


def test_as_of_prefers_current_unfinished_gw():
    boot = {
        "events": [
            {"id": 1, "deadline_time": "2026-08-21T17:30:00Z", "finished": True, "is_current": False, "is_next": False},
            {"id": 2, "deadline_time": "2026-08-28T17:30:00Z", "finished": False, "is_current": True, "is_next": False},
            {"id": 3, "deadline_time": "2026-09-04T17:30:00Z", "finished": False, "is_current": False, "is_next": True},
        ]
    }
    assert next_as_of_gw(boot) == 2


def test_gw_rows_skip_unfinished_fixture():
    elements = pd.DataFrame({"fpl_id": [10], "team_id": [1], "now_cost": [100], "selected_by_percent": [12.5]})
    fixtures = pd.DataFrame(
        {
            "fixture_id": [5, 6],
            "event": [2, 2],
            "team_h": [1, 3],
            "team_a": [2, 1],
            "finished": [1, 0],
            "kickoff_time": ["t", "t"],
        }
    )
    live = {
        "elements": [
            {
                "id": 10,
                "stats": {"minutes": 90, "total_points": 2},
                "explain": [
                    {"fixture": 5, "stats": [{"identifier": "minutes", "value": 90}]},
                    {"fixture": 6, "stats": [{"identifier": "minutes", "value": 10}]},
                ],
            }
        ]
    }
    rows = gw_rows_from_live("2026-27", 2, live, fixtures, elements)
    assert len(rows) == 1
    assert rows[0]["fixture_id"] == 5
    assert rows[0]["minutes"] == 90
    assert rows[0]["was_home"] == 1


def test_gw_rows_include_scored_unfinished_fixture():
    from fpl_model.ingest.fpl_api import fixture_is_done, gw_rows_from_live

    elements = pd.DataFrame({"fpl_id": [10], "team_id": [1], "now_cost": [100], "selected_by_percent": [12.5]})
    fixtures = pd.DataFrame(
        {
            "fixture_id": [6],
            "event": [2],
            "team_h": [1],
            "team_a": [2],
            "finished": [0],
            "team_h_score": [0],
            "team_a_score": [2],
            "kickoff_time": ["2026-09-12T14:00:00Z"],
        }
    )
    assert fixture_is_done(fixtures.iloc[0])
    live = {
        "elements": [
            {
                "id": 10,
                "stats": {"minutes": 90, "total_points": 2},
                "explain": [{"fixture": 6, "stats": [{"identifier": "minutes", "value": 90}]}],
            }
        ]
    }
    rows = gw_rows_from_live("2026-27", 2, live, fixtures, elements)
    assert len(rows) == 1
    assert rows[0]["fixture_id"] == 6


def test_fixture_is_done_past_kickoff():
    from fpl_model.ingest.fpl_api import fixture_is_done

    past = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    row = pd.Series(
        {
            "finished": 0,
            "team_h_score": None,
            "team_a_score": None,
            "kickoff_time": past,
        }
    )
    assert fixture_is_done(row)
    future = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    row2 = pd.Series({"finished": 0, "team_h_score": None, "team_a_score": None, "kickoff_time": future})
    assert not fixture_is_done(row2)
