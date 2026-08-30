from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd

from fpl_model.config import Settings, load_settings
from fpl_model.db import connect, init_db

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD", "GK": "GK", "GKP": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}

GW_COLS = {
    "element": "fpl_id",
    "round": "round",
    "fixture": "fixture_id",
    "opponent_team": "opponent_team",
    "was_home": "was_home",
    "minutes": "minutes",
    "starts": "starts",
    "total_points": "total_points",
    "goals_scored": "goals_scored",
    "assists": "assists",
    "bonus": "bonus",
    "bps": "bps",
    "clean_sheets": "clean_sheets",
    "goals_conceded": "goals_conceded",
    "saves": "saves",
    "expected_goals": "expected_goals",
    "expected_assists": "expected_assists",
    "expected_goal_involvements": "expected_goal_involvements",
    "expected_goals_conceded": "expected_goals_conceded",
    "value": "value",
    "selected": "selected",
    "defensive_contribution": "defensive_contribution",
    "tackles": "tackles",
    "recoveries": "recoveries",
    "clearances_blocks_interceptions": "clearances_blocks_interceptions",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "penalties_missed": "penalties_missed",
    "penalties_saved": "penalties_saved",
    "threat": "threat",
    "creativity": "creativity",
    "influence": "influence",
    "ict_index": "ict_index",
    "kickoff_time": "kickoff_time",
}


def _get(client: httpx.Client, url: str) -> bytes | None:
    r = client.get(url)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.content


def _read_csv_bytes(content: bytes) -> pd.DataFrame:
    return pd.read_csv(pd.io.common.BytesIO(content))


def download_season(season: str, settings: Settings | None = None, client: httpx.Client | None = None) -> Path:
    settings = settings or load_settings()
    dest = settings.raw_dir / season
    dest.mkdir(parents=True, exist_ok=True)
    base = settings.vaastav_base_url
    files = {
        "merged_gw.csv": f"{base}/data/{season}/gws/merged_gw.csv",
        "fixtures.csv": f"{base}/data/{season}/fixtures.csv",
        "teams.csv": f"{base}/data/{season}/teams.csv",
        "players_raw.csv": f"{base}/data/{season}/players_raw.csv",
        "id_dict.csv": f"{base}/data/{season}/id_dict.csv",
    }
    own = client is None
    client = client or httpx.Client(timeout=60.0, follow_redirects=True)
    try:
        for name, url in files.items():
            path = dest / name
            if path.exists() and path.stat().st_size > 0:
                continue
            content = _get(client, url)
            if content is None and name == "id_dict.csv":
                content = _get(client, f"{base}/data/id_dict.csv")
            if content is None:
                continue
            path.write_bytes(content)
    finally:
        if own:
            client.close()
    return dest


def _norm_position(val) -> str | None:
    if pd.isna(val):
        return None
    if isinstance(val, str) and val.strip().isdigit():
        val = int(val)
    return POSITION_MAP.get(val, str(val).upper()[:3] if isinstance(val, str) else None)


def _to_sql(conn, df: pd.DataFrame, table: str) -> None:
    df.to_sql(table, conn, if_exists="append", index=False)


def _clear_season(conn, season: str) -> None:
    for table in (
        "score_player_asof",
        "feat_player_asof",
        "fact_fpl_player_gw",
        "fact_fixture",
        "map_player_ids",
        "dim_player",
        "dim_team",
    ):
        conn.execute(f"DELETE FROM {table} WHERE season = ?", (season,))
    conn.execute("INSERT OR IGNORE INTO dim_season(season) VALUES (?)", (season,))


def load_season(season: str, settings: Settings | None = None) -> None:
    settings = settings or load_settings()
    dest = settings.raw_dir / season
    if not (dest / "merged_gw.csv").exists():
        download_season(season, settings)

    conn = connect(settings)
    init_db(conn, settings)
    _clear_season(conn, season)

    teams = pd.read_csv(dest / "teams.csv")
    dim_team = pd.DataFrame(
        {
            "season": season,
            "team_id": teams["id"],
            "name": teams["name"],
            "short_name": teams.get("short_name", teams["name"]),
        }
    )
    _to_sql(conn, dim_team, "dim_team")

    players = pd.read_csv(dest / "players_raw.csv")
    pos = players["element_type"].map(_norm_position)
    dim_player = pd.DataFrame(
        {
            "season": season,
            "fpl_id": players["id"],
            "web_name": players.get("web_name", players.get("second_name")),
            "first_name": players.get("first_name"),
            "second_name": players.get("second_name"),
            "position": pos,
            "team_id": players.get("team"),
            "status": players.get("status"),
            "chance_of_playing_this": pd.to_numeric(players.get("chance_of_playing_this_round"), errors="coerce"),
            "chance_of_playing_next": pd.to_numeric(players.get("chance_of_playing_next_round"), errors="coerce"),
            "penalties_order": pd.to_numeric(players.get("penalties_order"), errors="coerce"),
            "corners_order": pd.to_numeric(players.get("corners_and_indirect_freekicks_order"), errors="coerce"),
            "direct_fk_order": pd.to_numeric(players.get("direct_freekicks_order"), errors="coerce"),
            "now_cost": players.get("now_cost"),
        }
    )
    _to_sql(conn, dim_player, "dim_player")

    gw = pd.read_csv(dest / "merged_gw.csv")
    out = pd.DataFrame({"season": season})
    for src, dst in GW_COLS.items():
        if src in gw.columns:
            out[dst] = gw[src]
        else:
            out[dst] = pd.NA
    if "position" in gw.columns:
        gw_pos = gw["position"].map(_norm_position)
        # fill dim later if needed
        _ = gw_pos
    numeric = [
        "fpl_id",
        "round",
        "fixture_id",
        "opponent_team",
        "minutes",
        "starts",
        "total_points",
        "goals_scored",
        "assists",
        "bonus",
        "bps",
        "clean_sheets",
        "goals_conceded",
        "saves",
        "value",
        "selected",
        "tackles",
        "recoveries",
        "clearances_blocks_interceptions",
        "yellow_cards",
        "red_cards",
        "penalties_missed",
        "penalties_saved",
    ]
    for c in numeric:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in [
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
        "defensive_contribution",
        "threat",
        "creativity",
        "influence",
        "ict_index",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["was_home"] = out["was_home"].map(lambda x: 1 if str(x).lower() in {"true", "1"} else (0 if str(x).lower() in {"false", "0"} else pd.NA))
    out["starts"] = out["starts"].fillna(0)
    out["fixture_id"] = out["fixture_id"].fillna(-1)
    _to_sql(conn, out, "fact_fpl_player_gw")

    fx = pd.read_csv(dest / "fixtures.csv")
    fact_fx = pd.DataFrame(
        {
            "season": season,
            "fixture_id": fx["id"],
            "event": pd.to_numeric(fx.get("event"), errors="coerce"),
            "team_h": fx["team_h"],
            "team_a": fx["team_a"],
            "team_h_score": pd.to_numeric(fx.get("team_h_score"), errors="coerce"),
            "team_a_score": pd.to_numeric(fx.get("team_a_score"), errors="coerce"),
            "finished": fx.get("finished").map(lambda x: 1 if str(x).lower() in {"true", "1"} else 0) if "finished" in fx.columns else 0,
            "kickoff_time": fx.get("kickoff_time"),
            "team_h_difficulty": fx.get("team_h_difficulty"),
            "team_a_difficulty": fx.get("team_a_difficulty"),
        }
    )
    _to_sql(conn, fact_fx, "fact_fixture")

    id_path = dest / "id_dict.csv"
    if id_path.exists():
        ids = pd.read_csv(id_path)
        cols = {c.lower(): c for c in ids.columns}
        fpl_col = cols.get("fpl_id") or cols.get("fpl id")
        us_col = cols.get("understat_id")
        mapping = pd.DataFrame(
            {
                "season": season,
                "fpl_id": pd.to_numeric(ids[fpl_col], errors="coerce") if fpl_col else pd.NA,
                "understat_id": pd.to_numeric(ids[us_col], errors="coerce") if us_col else pd.NA,
                "fpl_name": ids[cols["fpl_name"]] if "fpl_name" in cols else pd.NA,
                "understat_name": ids[cols["understat_name"]] if "understat_name" in cols else pd.NA,
            }
        )
        _to_sql(conn, mapping, "map_player_ids")

    conn.commit()
    conn.close()
