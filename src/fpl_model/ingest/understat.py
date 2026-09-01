from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from fpl_model.config import Settings, load_settings
from fpl_model.ingest.id_map import map_players, team_key

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
BOX_X = 0.83
BOX_Y0, BOX_Y1 = 0.21, 0.79


def understat_year(season: str) -> str:
    return season.split("-")[0]


def _cache_dir(settings: Settings, season: str) -> Path:
    p = settings.data_dir / "raw" / "understat" / season
    p.mkdir(parents=True, exist_ok=True)
    (p / "players").mkdir(exist_ok=True)
    return p


def _client() -> httpx.Client:
    headers = {"User-Agent": UA, "X-Requested-With": "XMLHttpRequest", "Referer": "https://understat.com/"}
    return httpx.Client(timeout=60.0, follow_redirects=True, headers=headers)


def download_league(season: str, settings: Settings | None = None, force: bool = False) -> dict:
    settings = settings or load_settings()
    year = understat_year(season)
    dest = _cache_dir(settings, season) / "league.json"
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return json.loads(dest.read_text())
    with _client() as client:
        client.get(f"https://understat.com/league/EPL/{year}")
        r = client.get(f"https://understat.com/getLeagueData/EPL/{year}")
        r.raise_for_status()
        data = r.json()
    dest.write_text(json.dumps(data))
    return data


def download_player(
    understat_id: int,
    season: str,
    settings: Settings | None = None,
    force: bool = False,
    client: httpx.Client | None = None,
) -> dict | None:
    settings = settings or load_settings()
    dest = _cache_dir(settings, season) / "players" / f"{understat_id}.json"
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return json.loads(dest.read_text())
    own = client is None
    client = client or _client()
    try:
        r = client.get(f"https://understat.com/getPlayerData/{understat_id}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
    finally:
        if own:
            client.close()
    dest.write_text(json.dumps(data))
    return data


def download_mapped_players(
    understat_ids: list[int],
    season: str,
    settings: Settings | None = None,
    pause: float = 0.12,
    force: bool = False,
) -> int:
    settings = settings or load_settings()
    n = 0
    with _client() as client:
        client.get("https://understat.com/league/EPL/" + understat_year(season))
        for i, uid in enumerate(understat_ids):
            dest = _cache_dir(settings, season) / "players" / f"{uid}.json"
            cached = dest.exists() and dest.stat().st_size > 0 and not force
            data = download_player(uid, season, settings, force=force, client=client)
            if data:
                n += 1
            if pause and not cached and i + 1 < len(understat_ids):
                time.sleep(pause)
    return n


SET_PIECE_SIT = frozenset({"FromCorner", "SetPiece", "DirectFreekick"})


def shot_xg_by_match(shots: list, us_year: str) -> tuple[dict[str, float], dict[str, float]]:
    """Non-penalty open-play xG and set-piece (corner/FK) xG by match_id."""
    open_xg: dict[str, float] = defaultdict(float)
    sp_xg: dict[str, float] = defaultdict(float)
    for s in shots or []:
        if str(s.get("season")) != us_year:
            continue
        mid = str(s.get("match_id") or "")
        if not mid:
            continue
        try:
            xg = float(s.get("xG") or 0.0)
        except (TypeError, ValueError):
            continue
        sit = s.get("situation") or "OpenPlay"
        if sit == "Penalty":
            continue
        if sit == "OpenPlay":
            open_xg[mid] += xg
        else:
            sp_xg[mid] += xg
    return dict(open_xg), dict(sp_xg)


def _shot_in_box(shot: dict) -> bool:
    try:
        x = float(shot.get("X") or 0)
        y = float(shot.get("Y") or 0)
    except (TypeError, ValueError):
        return False
    return x >= BOX_X and BOX_Y0 <= y <= BOX_Y1


def matches_from_player_payload(payload: dict, us_year: str) -> pd.DataFrame:
    uid = int(payload["player"]["id"])
    shots = [s for s in payload.get("shots") or [] if str(s.get("season")) == us_year]
    box_by_match: dict[str, int] = {}
    for s in shots:
        mid = str(s.get("match_id"))
        box_by_match[mid] = box_by_match.get(mid, 0) + int(_shot_in_box(s))
    open_xg, sp_xg = shot_xg_by_match(payload.get("shots") or [], us_year)
    rows = []
    for m in payload.get("matches") or []:
        if str(m.get("season")) != us_year:
            continue
        mid = str(m.get("id"))
        rows.append(
            {
                "understat_id": uid,
                "match_id": int(m["id"]) if m.get("id") else None,
                "date": str(m.get("date"))[:10],
                "minutes": pd.to_numeric(m.get("time"), errors="coerce"),
                "position": m.get("position"),
                "shots": pd.to_numeric(m.get("shots"), errors="coerce"),
                "shots_in_box": box_by_match.get(mid, 0),
                "key_passes": pd.to_numeric(m.get("key_passes"), errors="coerce"),
                "goals": pd.to_numeric(m.get("goals"), errors="coerce"),
                "assists": pd.to_numeric(m.get("assists"), errors="coerce"),
                "xg": pd.to_numeric(m.get("xG"), errors="coerce"),
                "xa": pd.to_numeric(m.get("xA"), errors="coerce"),
                "npg": pd.to_numeric(m.get("npg"), errors="coerce"),
                "npxg": pd.to_numeric(m.get("npxG"), errors="coerce"),
                "xgchain": pd.to_numeric(m.get("xGChain"), errors="coerce"),
                "xgbuildup": pd.to_numeric(m.get("xGBuildup"), errors="coerce"),
                "open_npxg": open_xg.get(mid, 0.0),
                "sp_xg": sp_xg.get(mid, 0.0),
                "h_team": m.get("h_team"),
                "a_team": m.get("a_team"),
            }
        )
    return pd.DataFrame(rows)


def load_shot_splits(season: str, settings: Settings | None = None) -> pd.DataFrame:
    """Open-play / set-piece xG by understat player-match from cached player JSON."""
    settings = settings or load_settings()
    year = understat_year(season)
    cache = _cache_dir(settings, season) / "players"
    rows: list[dict] = []
    if not cache.exists():
        return pd.DataFrame(columns=["understat_id", "match_id", "open_npxg", "sp_xg"])
    for path in cache.glob("*.json"):
        try:
            payload = json.loads(path.read_text())
            uid = int(payload["player"]["id"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        open_xg, sp_xg = shot_xg_by_match(payload.get("shots") or [], year)
        mids = set(open_xg) | set(sp_xg)
        for mid in mids:
            try:
                mid_i = int(mid)
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "understat_id": uid,
                    "match_id": mid_i,
                    "open_npxg": open_xg.get(mid, 0.0),
                    "sp_xg": sp_xg.get(mid, 0.0),
                }
            )
    return pd.DataFrame(rows)


def team_matches_from_league(league: dict) -> pd.DataFrame:
    rows = []
    for d in league.get("dates") or []:
        hid = int(d["h"]["id"])
        aid = int(d["a"]["id"])
        htitle = d["h"]["title"]
        atitle = d["a"]["title"]
        xgh = pd.to_numeric(d.get("xG", {}).get("h"), errors="coerce") if isinstance(d.get("xG"), dict) else None
        xga = pd.to_numeric(d.get("xG", {}).get("a"), errors="coerce") if isinstance(d.get("xG"), dict) else None
        gh = pd.to_numeric((d.get("goals") or {}).get("h"), errors="coerce")
        ga = pd.to_numeric((d.get("goals") or {}).get("a"), errors="coerce")
        dt = str(d.get("datetime") or "")[:19]
        finished = 1 if d.get("isResult") else 0
        rows.append(
            {
                "match_id": int(d["id"]),
                "datetime": dt,
                "date": dt[:10],
                "finished": finished,
                "team_title": htitle,
                "team_key": team_key(htitle),
                "understat_team_id": hid,
                "is_home": 1,
                "xg_for": xgh,
                "xg_against": xga,
                "gf": gh,
                "ga": ga,
            }
        )
        rows.append(
            {
                "match_id": int(d["id"]),
                "datetime": dt,
                "date": dt[:10],
                "finished": finished,
                "team_title": atitle,
                "team_key": team_key(atitle),
                "understat_team_id": aid,
                "is_home": 0,
                "xg_for": xga,
                "xg_against": xgh,
                "gf": ga,
                "ga": gh,
            }
        )
    return pd.DataFrame(rows)


def fpl_frame_for_map(conn, season: str) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT p.fpl_id, p.first_name, p.second_name, p.web_name, t.name AS team_name
        FROM dim_player p
        LEFT JOIN dim_team t ON p.season = t.season AND p.team_id = t.team_id
        WHERE p.season = ?
        """,
        conn,
        params=(season,),
    )


def build_and_store_map(conn, season: str, league: dict) -> pd.DataFrame:
    players = pd.DataFrame(league["players"])
    players = players.rename(columns={"id": "understat_id"})
    fpl = fpl_frame_for_map(conn, season)
    mapping = map_players(fpl, players)
    mapping["season"] = season
    conn.execute("DELETE FROM map_player_ids WHERE season = ?", (season,))
    mapping.to_sql("map_player_ids", conn, if_exists="append", index=False)
    conn.commit()
    return mapping


def map_coverage_summary(conn, season: str) -> dict[str, Any]:
    """Understat↔FPL coverage counts for CLI / update hygiene."""
    rows = pd.read_sql_query(
        """
        SELECT match_method, COUNT(*) AS n
        FROM map_player_ids
        WHERE season = ?
        GROUP BY match_method
        """,
        conn,
        params=(season,),
    )
    by = {str(r["match_method"]): int(r["n"]) for _, r in rows.iterrows()} if not rows.empty else {}
    n_fpl = int(
        pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM dim_player WHERE season = ?",
            conn,
            params=(season,),
        )["n"].iloc[0]
    )
    matched = sum(v for k, v in by.items() if k not in {"unmatched_fpl", "unmatched_understat"})
    # Rows with both ids set
    both = pd.read_sql_query(
        """
        SELECT COUNT(*) AS n FROM map_player_ids
        WHERE season = ? AND fpl_id IS NOT NULL AND understat_id IS NOT NULL
        """,
        conn,
        params=(season,),
    )["n"].iloc[0]
    unmatched_fpl = int(by.get("unmatched_fpl", 0))
    return {
        "season": season,
        "fpl_players": n_fpl,
        "matched": int(both),
        "matched_pct": round(100.0 * float(both) / n_fpl, 1) if n_fpl else 0.0,
        "unmatched_fpl": unmatched_fpl,
        "unmatched_understat": int(by.get("unmatched_understat", 0)),
        "by_method": by,
    }


def write_id_map_review(conn, season: str, path: Path | None = None) -> Path:
    from fpl_model.config import load_settings

    settings = load_settings()
    dest = path or (settings.data_dir / f"id_map_{season}.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_sql_query(
        """
        SELECT
            m.match_method,
            m.fpl_id,
            m.fpl_name,
            p.web_name,
            t.name AS team,
            m.understat_id,
            m.understat_name
        FROM map_player_ids m
        LEFT JOIN dim_player p ON p.season = m.season AND p.fpl_id = m.fpl_id
        LEFT JOIN dim_team t ON t.season = p.season AND t.team_id = p.team_id
        WHERE m.season = ?
        ORDER BY
            CASE m.match_method
                WHEN 'unmatched_understat' THEN 0
                WHEN 'unmatched_fpl' THEN 1
                WHEN 'manual_alias' THEN 2
                ELSE 3
            END,
            COALESCE(m.fpl_name, m.understat_name)
        """,
        conn,
        params=(season,),
    )
    df.to_csv(dest, index=False)
    return dest


def store_understat_facts(
    conn, season: str, league: dict, settings: Settings | None = None, force: bool = False
) -> tuple[int, int]:
    settings = settings or load_settings()
    year = understat_year(season)
    conn.execute("DELETE FROM fact_understat_player_match WHERE season = ?", (season,))
    conn.execute("DELETE FROM fact_understat_team_match WHERE season = ?", (season,))

    tm = team_matches_from_league(league)
    tm["season"] = season
    tm.to_sql("fact_understat_team_match", conn, if_exists="append", index=False)

    mapped = pd.read_sql_query(
        "SELECT DISTINCT understat_id FROM map_player_ids WHERE season = ? AND understat_id IS NOT NULL",
        conn,
        params=(season,),
    )
    ids = [int(x) for x in mapped["understat_id"].dropna().tolist()]
    n_players = download_mapped_players(ids, season, settings, force=force)
    frames = []
    cache = _cache_dir(settings, season) / "players"
    for uid in ids:
        path = cache / f"{uid}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        mf = matches_from_player_payload(payload, year)
        if not mf.empty:
            frames.append(mf)
    if frames:
        pm = pd.concat(frames, ignore_index=True)
        pm["season"] = season
        pm.to_sql("fact_understat_player_match", conn, if_exists="append", index=False)
        n_matches = len(pm)
    else:
        n_matches = 0
    conn.commit()
    return n_players, n_matches
