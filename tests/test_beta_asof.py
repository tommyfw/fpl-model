import sqlite3

from fpl_model.db import init_db
from fpl_model.ingest.fpl_api import beta_as_of_gw, round_results_ready
from fpl_model.features.asof import asof_gw_frame
import pandas as pd


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    conn.execute("INSERT INTO dim_season(season) VALUES ('2026-27')")
    for i in range(1, 11):
        conn.execute(
            """INSERT INTO fact_fixture(season, fixture_id, event, finished)
               VALUES ('2026-27', ?, 2, 1)""",
            (i,),
        )
    conn.execute(
        """INSERT INTO fact_fpl_player_gw(season, fpl_id, round, fixture_id, minutes)
           VALUES ('2026-27', 1, 2, 1, 90)"""
    )
    conn.commit()
    return conn


def test_beta_when_current_gw_results_are_in():
    conn = _conn()
    assert round_results_ready(conn, "2026-27", 2)
    assert beta_as_of_gw(conn, "2026-27", 2) == 3


def test_no_beta_if_fixtures_open():
    conn = _conn()
    conn.execute("UPDATE fact_fixture SET finished = 0 WHERE fixture_id = 10")
    conn.commit()
    assert not round_results_ready(conn, "2026-27", 2)
    assert beta_as_of_gw(conn, "2026-27", 2) is None


def test_beta_asof_features_include_current_gw_results():
    gw = pd.DataFrame({"fpl_id": [1, 1], "round": [1, 2], "minutes": [90, 90]})
    assert set(asof_gw_frame(gw, 2)["round"]) == {1}
    assert set(asof_gw_frame(gw, 3)["round"]) == {1, 2}
