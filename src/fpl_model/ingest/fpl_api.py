from __future__ import annotations

import json
from pathlib import Path

import httpx
import pandas as pd

from fpl_model.config import Settings, load_settings
from fpl_model.db import connect, init_db
from fpl_model.ingest.vaastav import GW_COLS, POSITION_MAP, _norm_position, _to_sql

FPL_API = "https://fantasy.premierleague.com/api"
UA = "Mozilla/5.0 (compatible; fpl-model/0.1; research)"

LIVE_TO_GW = {
    "minutes": "minutes",
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
    "starts": "starts",
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
}


def _client() -> httpx.Client:
    return httpx.Client(timeout=60.0, follow_redirects=True, headers={"User-Agent": UA})


def cache_dir(settings: Settings, season: str) -> Path:
    p = settings.data_dir / "raw" / "fpl_api" / season
    p.mkdir(parents=True, exist_ok=True)
    return p


def season_from_bootstrap(boot: dict) -> str:
    events = boot.get("events") or []
    if not events:
        raise ValueError("bootstrap-static has no events")
    dl = str(events[0].get("deadline_time") or "")
    y, month = int(dl[:4]), int(dl[5:7])
    if month >= 7:
        return f"{y}-{str(y + 1)[-2:]}"
    return f"{y - 1}-{str(y)[-2:]}"


def next_as_of_gw(boot: dict) -> int:
    """As-of T is the current unfinished GW (use only T-1 and earlier)."""
    events = boot.get("events") or []
    for e in events:
        if e.get("is_current"):
            return int(e["id"])
    for e in events:
        if e.get("is_next"):
            return int(e["id"])
    unfinished = [int(e["id"]) for e in events if not e.get("finished")]
    if unfinished:
        return min(unfinished)
    return max(int(e["id"]) for e in events)


def round_results_ready(conn, season: str, gw: int) -> bool:
    """True when every fixture in ``gw`` is finished and player-GW rows exist."""
    fx = conn.execute(
        """
        SELECT COUNT(*) AS n,
               SUM(CASE WHEN finished = 1 THEN 1 ELSE 0 END) AS done
        FROM fact_fixture
        WHERE season = ? AND event = ?
        """,
        (season, int(gw)),
    ).fetchone()
    n = int(fx["n"] or 0) if fx is not None else 0
    done = int(fx["done"] or 0) if fx is not None else 0
    if n <= 0 or done < n:
        return False
    players = conn.execute(
        "SELECT COUNT(*) AS n FROM fact_fpl_player_gw WHERE season = ? AND round = ?",
        (season, int(gw)),
    ).fetchone()
    return int(players["n"] or 0) > 0


def beta_as_of_gw(conn, season: str, official_as_of: int) -> int | None:
    """If FPL current is T but GW T results are in, return T+1 for a beta snapshot."""
    t = int(official_as_of)
    if t < 1:
        return None
    if round_results_ready(conn, season, t):
        return t + 1
    return None


def fetch_bootstrap(settings: Settings | None = None, season: str | None = None, force: bool = False) -> dict:
    settings = settings or load_settings()
    dest_guess = cache_dir(settings, season or "live") / "bootstrap-static.json"
    if dest_guess.exists() and dest_guess.stat().st_size > 0 and not force:
        return json.loads(dest_guess.read_text())
    with _client() as client:
        r = client.get(f"{FPL_API}/bootstrap-static/")
        r.raise_for_status()
        boot = r.json()
    inferred = season_from_bootstrap(boot)
    dest = cache_dir(settings, inferred) / "bootstrap-static.json"
    dest.write_text(json.dumps(boot))
    return boot


def fetch_fixtures(season: str, settings: Settings | None = None, force: bool = False) -> list:
    settings = settings or load_settings()
    dest = cache_dir(settings, season) / "fixtures.json"
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return json.loads(dest.read_text())
    with _client() as client:
        r = client.get(f"{FPL_API}/fixtures/")
        r.raise_for_status()
        data = r.json()
    dest.write_text(json.dumps(data))
    return data


def fetch_event_live(season: str, event: int, settings: Settings | None = None, force: bool = False) -> dict:
    settings = settings or load_settings()
    dest = cache_dir(settings, season) / "live" / f"event-{event}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return json.loads(dest.read_text())
    with _client() as client:
        r = client.get(f"{FPL_API}/event/{event}/live/")
        r.raise_for_status()
        data = r.json()
    dest.write_text(json.dumps(data))
    return data


def _entry_cache(settings: Settings, entry_id: int, name: str) -> Path:
    p = settings.data_dir / "raw" / "fpl_api" / "entries" / str(int(entry_id))
    p.mkdir(parents=True, exist_ok=True)
    return p / name


def fetch_entry(entry_id: int, settings: Settings | None = None, force: bool = False) -> dict:
    settings = settings or load_settings()
    dest = _entry_cache(settings, entry_id, "entry.json")
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return json.loads(dest.read_text())
    with _client() as client:
        r = client.get(f"{FPL_API}/entry/{int(entry_id)}/")
        r.raise_for_status()
        data = r.json()
    dest.write_text(json.dumps(data))
    return data


def fetch_entry_history(entry_id: int, settings: Settings | None = None, force: bool = False) -> dict:
    settings = settings or load_settings()
    dest = _entry_cache(settings, entry_id, "history.json")
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return json.loads(dest.read_text())
    with _client() as client:
        r = client.get(f"{FPL_API}/entry/{int(entry_id)}/history/")
        r.raise_for_status()
        data = r.json()
    dest.write_text(json.dumps(data))
    return data


def fetch_entry_picks(entry_id: int, event: int, settings: Settings | None = None, force: bool = False) -> dict:
    settings = settings or load_settings()
    dest = _entry_cache(settings, entry_id, f"picks-{int(event)}.json")
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return json.loads(dest.read_text())
    with _client() as client:
        r = client.get(f"{FPL_API}/entry/{int(entry_id)}/event/{int(event)}/picks/")
        r.raise_for_status()
        data = r.json()
    dest.write_text(json.dumps(data))
    return data


def estimate_free_transfers(history: dict) -> int:
    """Approximate FTs at the start of the next GW from public history (max 5)."""
    past = list(history.get("past") or [])
    chips = history.get("chips") or []
    chip_events = {
        int(c["event"])
        for c in chips
        if str(c.get("name") or "").lower() in {"wildcard", "freehit"} and c.get("event") is not None
    }
    ft = 1
    for row in sorted(past, key=lambda x: int(x.get("event") or 0)):
        ev = int(row.get("event") or 0)
        used = int(row.get("event_transfers") or 0)
        if ev in chip_events:
            ft = 1
            continue
        leftover = max(0, ft - used)
        ft = min(5, leftover + 1)
    return int(ft)


def parse_entry_team(entry: dict, history: dict, picks: dict) -> dict:
    """Public FPL team: 15 ids, bank £m, estimated FTs, chips already played."""
    ids = [int(p["element"]) for p in (picks.get("picks") or []) if p.get("element") is not None]
    bank_raw = entry.get("last_deadline_bank")
    if bank_raw is None and picks.get("entry_history"):
        bank_raw = (picks["entry_history"] or {}).get("bank")
    bank = float(bank_raw or 0) / 10.0
    chips = [str(c.get("name") or "").lower() for c in (history.get("chips") or [])]
    name = entry.get("name") or ""
    return {
        "ok": len(ids) == 15,
        "entry_id": entry.get("id"),
        "name": name,
        "player_name": f"{entry.get('player_first_name') or ''} {entry.get('player_last_name') or ''}".strip(),
        "ids": ids,
        "bank": round(bank, 1),
        "free_transfers": estimate_free_transfers(history),
        "chips_used": chips,
        "current_event": entry.get("current_event"),
        "error": None if len(ids) == 15 else "could not load a full 15 (picks may be hidden until the deadline)",
    }


def load_entry_team(entry_id: int, settings: Settings | None = None, force: bool = False) -> dict:
    settings = settings or load_settings()
    entry = fetch_entry(entry_id, settings, force=force)
    history = fetch_entry_history(entry_id, settings, force=force)
    event = entry.get("current_event") or ((history.get("current") or {}).get("event"))
    if not event:
        past = history.get("past") or []
        event = max((int(r["event"]) for r in past if r.get("event")), default=1)
    try:
        picks = fetch_entry_picks(entry_id, int(event), settings, force=force)
    except Exception:
        past = history.get("past") or []
        prev = max((int(r["event"]) for r in past if r.get("event")), default=None)
        if prev is None or prev == event:
            raise
        picks = fetch_entry_picks(entry_id, prev, settings, force=force)
    return parse_entry_team(entry, history, picks)


def _clear_fpl_facts(conn, season: str) -> None:
    for table in (
        "score_player_asof",
        "feat_player_asof",
        "fact_fpl_player_gw",
        "fact_fixture",
        "dim_player",
        "dim_team",
    ):
        conn.execute(f"DELETE FROM {table} WHERE season = ?", (season,))
    conn.execute("INSERT OR IGNORE INTO dim_season(season) VALUES (?)", (season,))


def _stats_row(stats: dict, extra: dict) -> dict:
    row = dict(extra)
    for src, dst in LIVE_TO_GW.items():
        row[dst] = stats.get(src)
    return row


def gw_rows_from_live(
    season: str,
    event: int,
    live: dict,
    fixtures: pd.DataFrame,
    elements: pd.DataFrame,
) -> list[dict]:
    """One row per player-fixture for a finished GW. Unfinished fixtures are skipped."""
    fx_e = fixtures.loc[fixtures["event"] == event]
    finished_ids = set(fx_e.loc[fx_e["finished"] == 1, "fixture_id"].tolist())
    team_of = dict(zip(elements["fpl_id"], elements["team_id"]))
    now_cost = dict(zip(elements["fpl_id"], elements["now_cost"]))
    selected = dict(zip(elements["fpl_id"], elements.get("selected_by_percent", pd.Series(dtype=float))))
    rows = []
    for el in live.get("elements") or []:
        fpl_id = el.get("id")
        team = team_of.get(fpl_id)
        explains = el.get("explain") or []
        if not explains:
            team_fx = fx_e.loc[(fx_e["team_h"] == team) | (fx_e["team_a"] == team)]
            explains = [{"fixture": int(r["fixture_id"]), "stats": []} for _, r in team_fx.iterrows()]
            if not explains:
                continue
        for part in explains:
            fxid = part.get("fixture")
            if fxid not in finished_ids:
                continue
            fr = fx_e.loc[fx_e["fixture_id"] == fxid]
            if fr.empty:
                continue
            fr = fr.iloc[0]
            was_home = 1 if int(fr["team_h"]) == team else 0
            opp = int(fr["team_a"] if was_home else fr["team_h"])
            stats = {s["identifier"]: s.get("value") for s in (part.get("stats") or []) if isinstance(s, dict) and "identifier" in s}
            if not stats:
                stats = el.get("stats") or {}
            live_stats = el.get("stats") or {}
            if not stats.get("total_points") and live_stats.get("total_points") is not None:
                stats = {**stats, "total_points": live_stats.get("total_points")}
            row = _stats_row(
                stats,
                {
                    "season": season,
                    "fpl_id": fpl_id,
                    "round": event,
                    "fixture_id": fxid,
                    "opponent_team": opp,
                    "was_home": was_home,
                    "kickoff_time": fr.get("kickoff_time"),
                    "value": now_cost.get(fpl_id),
                    "selected": selected.get(fpl_id),
                },
            )
            rows.append(row)
    return rows


def load_live_season(season: str | None = None, settings: Settings | None = None, force: bool = False) -> tuple[str, int]:
    """Replace FPL facts for the live season. Keeps Understat rows for that season."""
    settings = settings or load_settings()
    boot = fetch_bootstrap(settings, season, force=force)
    inferred = season_from_bootstrap(boot)
    season = season or inferred
    fx_raw = fetch_fixtures(season, settings, force=force)

    teams = pd.DataFrame(boot["teams"])
    elements = pd.DataFrame(boot["elements"])
    elements = elements.rename(columns={"id": "fpl_id", "team": "team_id", "element_type": "element_type"})
    elements["position"] = elements["element_type"].map(lambda x: POSITION_MAP.get(x, _norm_position(x)))
    elements["now_cost"] = pd.to_numeric(elements.get("now_cost"), errors="coerce")
    elements["selected_by_percent"] = pd.to_numeric(elements.get("selected_by_percent"), errors="coerce")

    fixtures = pd.DataFrame(
        {
            "season": season,
            "fixture_id": [f["id"] for f in fx_raw],
            "event": [f.get("event") for f in fx_raw],
            "team_h": [f.get("team_h") for f in fx_raw],
            "team_a": [f.get("team_a") for f in fx_raw],
            "team_h_score": [f.get("team_h_score") for f in fx_raw],
            "team_a_score": [f.get("team_a_score") for f in fx_raw],
            "finished": [1 if f.get("finished") else 0 for f in fx_raw],
            "kickoff_time": [f.get("kickoff_time") for f in fx_raw],
            "team_h_difficulty": [f.get("team_h_difficulty") for f in fx_raw],
            "team_a_difficulty": [f.get("team_a_difficulty") for f in fx_raw],
        }
    )
    fixtures["event"] = pd.to_numeric(fixtures["event"], errors="coerce")

    finished_events = sorted(
        {
            int(e["id"])
            for e in boot["events"]
            if e.get("finished") or (e.get("is_current") and e.get("data_checked"))
        }
    )
    # also include any fixture-finished events not marked on the event object
    fx_done = fixtures.loc[fixtures["finished"] == 1, "event"].dropna().astype(int).unique().tolist()
    events_to_load = sorted(set(finished_events) | set(int(x) for x in fx_done if pd.notna(x)))

    gw_rows: list[dict] = []
    for ev in events_to_load:
        live = fetch_event_live(season, ev, settings, force=force)
        gw_rows.extend(gw_rows_from_live(season, ev, live, fixtures, elements))

    conn = connect(settings)
    init_db(conn, settings)
    _clear_fpl_facts(conn, season)

    dim_team = pd.DataFrame(
        {"season": season, "team_id": teams["id"], "name": teams["name"], "short_name": teams["short_name"]}
    )
    _to_sql(conn, dim_team, "dim_team")

    dim_player = pd.DataFrame(
        {
            "season": season,
            "fpl_id": elements["fpl_id"],
            "web_name": elements.get("web_name"),
            "first_name": elements.get("first_name"),
            "second_name": elements.get("second_name"),
            "position": elements["position"],
            "team_id": elements["team_id"],
            "status": elements.get("status"),
            "chance_of_playing_this": pd.to_numeric(elements.get("chance_of_playing_this_round"), errors="coerce"),
            "chance_of_playing_next": pd.to_numeric(elements.get("chance_of_playing_next_round"), errors="coerce"),
            "news": elements.get("news"),
            "penalties_order": pd.to_numeric(elements.get("penalties_order"), errors="coerce"),
            "corners_order": pd.to_numeric(elements.get("corners_and_indirect_freekicks_order"), errors="coerce"),
            "direct_fk_order": pd.to_numeric(elements.get("direct_freekicks_order"), errors="coerce"),
            "now_cost": elements["now_cost"],
            "code": pd.to_numeric(elements.get("code"), errors="coerce"),
            "selected_by_percent": elements["selected_by_percent"],
        }
    )
    _to_sql(conn, dim_player, "dim_player")
    _to_sql(conn, fixtures, "fact_fixture")

    if gw_rows:
        out = pd.DataFrame(gw_rows)
        for c in GW_COLS.values():
            if c not in out.columns:
                out[c] = pd.NA
        out["starts"] = pd.to_numeric(out.get("starts"), errors="coerce").fillna(0)
        out["total_points"] = pd.to_numeric(out.get("total_points"), errors="coerce")
        out["fixture_id"] = out["fixture_id"].fillna(-1)
        out = out.drop_duplicates(subset=["season", "fpl_id", "round", "fixture_id"], keep="last")
        keep = ["season"] + [c for c in GW_COLS.values() if c in out.columns]
        _to_sql(conn, out[keep], "fact_fpl_player_gw")

    conn.commit()
    conn.close()
    return season, next_as_of_gw(boot)
