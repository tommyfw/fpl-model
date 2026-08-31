import json
import sqlite3

from fpl_model.db import init_db
from fpl_model.ui import queries


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    conn.execute("INSERT INTO dim_season(season) VALUES ('2026-27')")
    conn.execute(
        "INSERT INTO dim_team(season, team_id, name, short_name) VALUES ('2026-27', 1, 'Arsenal', 'ARS')"
    )
    conn.execute(
        """INSERT INTO dim_player(
            season, fpl_id, web_name, first_name, second_name, position, team_id,
            status, now_cost, selected_by_percent
        ) VALUES ('2026-27', 1, 'Haaland', 'Erling', 'Haaland', 'FWD', 1, 'a', 145, 62.3)"""
    )
    card = {
        "overall": 74.9,
        "upside": 55.0,
        "action": "BUY",
        "upside_parts": {"finishing": 60.0},
    }
    conn.execute(
        """INSERT INTO score_player_asof(
            season, as_of_gw, fpl_id, position, team_id, web_name,
            U, F, R, T, Q, M, O, V, overall, upside, xpts, action, card_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            "2026-27",
            2,
            1,
            "FWD",
            1,
            "Haaland",
            63.6,
            80.0,
            70.0,
            75.0,
            72.0,
            90.0,
            60.0,
            40.0,
            74.9,
            55.0,
            12.5,
            "BUY",
            json.dumps(card),
        ),
    )
    conn.execute(
        """INSERT INTO feat_player_asof(season, as_of_gw, fpl_id, position, team_id, price, features_json)
           VALUES ('2026-27', 2, 1, 'FWD', 1, 14.5, '{}')"""
    )
    conn.commit()
    return conn


def test_list_and_get_player():
    conn = _conn()
    snaps = queries.available_snapshots(conn)
    assert snaps == [{"season": "2026-27", "as_of_gw": 2, "n": 1}]
    rows = queries.list_players(conn, "2026-27", 2, q="haal")
    assert len(rows) == 1
    assert rows[0]["web_name"] == "Haaland"
    assert rows[0]["team"] == "ARS"
    assert rows[0]["price"] == 14.5
    p = queries.get_player(conn, "2026-27", 2, 1)
    assert p["card"]["action"] == "BUY"
    assert p["first_name"] == "Erling"
    empty = queries.list_players(conn, "2026-27", 2, position="GK")
    assert empty == []


def test_fixture_ticker_uses_pack_and_league_bands():
    conn = _conn()
    conn.execute(
        "INSERT INTO dim_team(season, team_id, name, short_name) VALUES ('2026-27', 2, 'Burnley', 'BUR')"
    )
    leaky = json.dumps(
        {
            "team_attack": 1.1,
            "team_defence": -1.8,
            "team_attack_home": 1.2,
            "team_attack_away": 1.0,
            "team_defence_home": -1.5,
            "team_defence_away": -2.0,
        }
    )
    tight = json.dumps(
        {
            "team_attack": 1.4,
            "team_defence": -0.9,
            "team_attack_home": 1.5,
            "team_attack_away": 1.2,
            "team_defence_home": -0.8,
            "team_defence_away": -1.0,
        }
    )
    conn.execute(
        "UPDATE feat_player_asof SET features_json = ? WHERE fpl_id = 1",
        (tight,),
    )
    conn.execute(
        """INSERT INTO feat_player_asof(season, as_of_gw, fpl_id, position, team_id, price, features_json)
           VALUES ('2026-27', 2, 2, 'MID', 2, 5.0, ?)""",
        (leaky,),
    )
    conn.execute(
        """INSERT INTO fact_fixture(
            season, fixture_id, event, team_h, team_a, finished, team_h_difficulty, team_a_difficulty
        ) VALUES ('2026-27', 10, 2, 1, 2, 0, 2, 4)"""
    )
    conn.commit()
    out = queries.fixture_ticker(conn, "2026-27", 2)
    ars = next(t for t in out["teams"] if t["team_id"] == 1)
    fx = next(f for f in ars["fixtures"] if not f["blank"])
    assert abs(fx["leak"] - 2.0) < 1e-9
    assert fx["home"] is True
    assert fx["attack_ease"] > fx["leak"]  # strong home attack lifts ease above raw leak
    assert fx["attack_band"] == "easy"
    assert "ease" in fx["title"] and "leak" in fx["title"]
    assert fx["rating"] in (1, 2, 3, 4, 5)


def test_lineup_needs_fifteen():
    conn = _conn()
    out = queries.lineup_for_squad(conn, "2026-27", 2, [1])
    assert out["ok"] is False
    assert "15" in out["error"]



def test_player_stats_prefer_understat_totals():
    from fpl_model.config import load_settings

    feats = {
        "understat_matched": 1,
        "xg_source": "understat",
        "xg": 0,
        "xa": 0,
        "npxg": 0.75,
        "npxa": 0.0,
        "npxg90": 0.27,
        "npxa90": 0.10,
        "npxg90_obs": 0.75,
        "npxa90_obs": 0.0,
        "us_minutes": 90,
        "minutes": 90,
        "goals": 0,
        "assists": 0,
        "points": 2,
    }
    st = queries._player_stats(feats, settings=load_settings())
    assert st["xg_source"] == "understat"
    assert st["npxg"] == 0.75
    assert st["npxa"] == 0.0
    assert st["npxg90"] == 0.27
    assert st["rate_weight"] == 0.17
