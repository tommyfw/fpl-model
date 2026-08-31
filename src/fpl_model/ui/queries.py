from __future__ import annotations

import json
import sqlite3
from typing import Any

import pandas as pd

from fpl_model.config import Settings, load_settings
from fpl_model.features.asof import upcoming_fixtures
from fpl_model.squad.transfers import suggest_transfers


def available_snapshots(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT season, as_of_gw, COUNT(*) AS n
        FROM score_player_asof
        GROUP BY season, as_of_gw
        ORDER BY season DESC, as_of_gw DESC
        """
    ).fetchall()
    return [{"season": r["season"], "as_of_gw": int(r["as_of_gw"]), "n": int(r["n"])} for r in rows]


def default_snapshot(conn: sqlite3.Connection, preferred_season: str | None = None) -> dict[str, Any] | None:
    snaps = available_snapshots(conn)
    if not snaps:
        return None
    if preferred_season:
        for s in snaps:
            if s["season"] == preferred_season:
                return s
    return snaps[0]


def list_players(
    conn: sqlite3.Connection,
    season: str,
    as_of_gw: int,
    *,
    position: str | None = None,
    q: str | None = None,
    action: str | None = None,
    sort: str = "overall",
    limit: int = 400,
) -> list[dict[str, Any]]:
    sort_map = {
        "overall": "s.overall",
        "upside": "s.upside",
        "xpts": "s.xpts",
        "value": "s.V",
        "quality": "s.Q",
        "minutes": "s.M",
        "fixtures": "s.O",
        "price": "COALESCE(f.price, p.now_cost / 10.0)",
    }
    order = sort_map.get(sort, "s.overall")
    where = ["s.season = ?", "s.as_of_gw = ?"]
    params: list[Any] = [season, as_of_gw]
    if position:
        where.append("s.position = ?")
        params.append(position.upper())
    if action:
        where.append("s.action = ?")
        params.append(action)
    if q:
        where.append("LOWER(s.web_name) LIKE ?")
        params.append(f"%{q.lower()}%")
    sql = f"""
        SELECT
            s.fpl_id, s.web_name, s.position, s.team_id,
            t.short_name AS team, t.name AS team_name,
            p.status, p.selected_by_percent,
            COALESCE(f.price, p.now_cost / 10.0) AS price,
            s.overall, s.upside, s.xpts, s.action,
            s.Q, s.U, s.F, s.R, s.T, s.M, s.O, s.V
        FROM score_player_asof s
        LEFT JOIN dim_team t ON t.season = s.season AND t.team_id = s.team_id
        LEFT JOIN dim_player p ON p.season = s.season AND p.fpl_id = s.fpl_id
        LEFT JOIN feat_player_asof f
            ON f.season = s.season AND f.as_of_gw = s.as_of_gw AND f.fpl_id = s.fpl_id
        WHERE {" AND ".join(where)}
        ORDER BY {order} DESC, s.web_name
        LIMIT ?
    """
    params.append(int(limit))
    rows = conn.execute(sql, params).fetchall()
    return [_player_row(r) for r in rows]


def get_player(conn: sqlite3.Connection, season: str, as_of_gw: int, fpl_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT
            s.fpl_id, s.web_name, s.position, s.team_id, s.card_json,
            t.short_name AS team, t.name AS team_name,
            p.status, p.selected_by_percent, p.first_name, p.second_name,
            p.penalties_order, p.corners_order,
            COALESCE(f.price, p.now_cost / 10.0) AS price,
            f.features_json,
            s.overall, s.upside, s.xpts, s.action,
            s.Q, s.U, s.F, s.R, s.T, s.M, s.O, s.V
        FROM score_player_asof s
        LEFT JOIN dim_team t ON t.season = s.season AND t.team_id = s.team_id
        LEFT JOIN dim_player p ON p.season = s.season AND p.fpl_id = s.fpl_id
        LEFT JOIN feat_player_asof f
            ON f.season = s.season AND f.as_of_gw = s.as_of_gw AND f.fpl_id = s.fpl_id
        WHERE s.season = ? AND s.as_of_gw = ? AND s.fpl_id = ?
        """,
        (season, as_of_gw, fpl_id),
    ).fetchone()
    if row is None:
        return None
    out = _player_row(row)
    card = json.loads(row["card_json"]) if row["card_json"] else {}
    out["card"] = card
    out["first_name"] = row["first_name"]
    out["second_name"] = row["second_name"]
    out["penalties_order"] = row["penalties_order"]
    out["corners_order"] = row["corners_order"]
    feats = json.loads(row["features_json"]) if row["features_json"] else {}
    settings = load_settings()
    feats = _ensure_npx_totals(conn, season, as_of_gw, int(row["fpl_id"]), feats, settings)
    out["stats"] = _player_stats(feats, settings=settings)
    out["fixtures"] = player_fixtures(
        conn, season, as_of_gw, int(row["team_id"]) if row["team_id"] is not None else None, settings
    )
    return out


def _ensure_npx_totals(
    conn: sqlite3.Connection,
    season: str,
    as_of_gw: int,
    fpl_id: int,
    feats: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    """Backfill Understat season totals for older feature rows that only stored rates."""
    matched = int(feats.get("understat_matched") or 0) == 1
    if not matched or feats.get("npxg") is not None:
        return feats
    from fpl_model.features.asof import _asof_cutoff, load_fixtures

    fx = load_fixtures(conn, season)
    cutoff = _asof_cutoff(fx, as_of_gw)
    if cutoff is None:
        return feats
    cut_date = (
        cutoff.tz_convert("UTC").strftime("%Y-%m-%d")
        if getattr(cutoff, "tzinfo", None)
        else cutoff.strftime("%Y-%m-%d")
    )
    row = conn.execute(
        """
        SELECT SUM(u.npxg) AS npxg, SUM(u.xa) AS npxa, SUM(u.minutes) AS us_minutes
        FROM fact_understat_player_match u
        JOIN map_player_ids m
          ON m.season = u.season AND m.understat_id = u.understat_id
        WHERE u.season = ? AND m.fpl_id = ? AND u.date < ?
        """,
        (season, fpl_id, cut_date),
    ).fetchone()
    if row is None:
        return feats
    out = dict(feats)
    if row["npxg"] is not None:
        out["npxg"] = float(row["npxg"])
    if row["npxa"] is not None:
        out["npxa"] = float(row["npxa"])
    if row["us_minutes"] is not None:
        out["us_minutes"] = float(row["us_minutes"])
        m = float(row["us_minutes"])
        if m > 0:
            out.setdefault("npxg90_obs", float(row["npxg"] or 0) * 90.0 / m)
            out.setdefault("npxa90_obs", float(row["npxa"] or 0) * 90.0 / m)
    out["xg_source"] = "understat"
    return out


def _player_stats(feats: dict[str, Any], settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()

    def num(key: str, nd: int = 0) -> float | None:
        v = feats.get(key)
        if v is None:
            return None
        try:
            x = float(v)
        except (TypeError, ValueError):
            return None
        if pd.isna(x):
            return None
        return round(x, nd)

    matched = int(feats.get("understat_matched") or 0) == 1 or feats.get("xg_source") == "understat"
    source = "understat" if matched else "fpl"
    # Prefer explicit Understat/FPL npx totals; never mix null FPL xG with US rates.
    npxg = num("npxg", 2)
    npxa = num("npxa", 2)
    if npxg is None and not matched:
        npxg = num("xg", 2)
    if npxa is None and not matched:
        npxa = num("xa", 2)

    us_min = num("us_minutes", 0)
    sample_min = us_min if matched and us_min is not None else num("minutes", 0)
    prior = float(settings.shrink_prior_minutes)
    rate_weight = None
    if sample_min is not None and sample_min > 0:
        rate_weight = round(sample_min / (sample_min + prior), 2)

    return {
        "minutes": num("minutes", 0),
        "starts": num("starts", 0),
        "points": num("points", 0),
        "goals": num("goals", 0),
        "assists": num("assists", 0),
        "bonus": num("bonus", 0),
        "bps": num("bps", 0),
        "cs": num("cs", 0),
        "saves": num("saves", 0),
        "xg_source": source,
        "npxg": npxg,
        "npxa": npxa,
        "npxg90": num("npxg90", 2),
        "npxa90": num("npxa90", 2),
        "npxg90_obs": num("npxg90_obs", 2),
        "npxa90_obs": num("npxa90_obs", 2),
        "rate_weight": rate_weight,
        "defcon_hit_rate": num("defcon_hit_rate", 2),
        "appearances": num("appearances", 0),
    }


def _team_maps(conn: sqlite3.Connection, season: str) -> tuple[dict[int, str], dict[int, str]]:
    rows = conn.execute(
        "SELECT team_id, short_name, name FROM dim_team WHERE season = ?",
        (season,),
    ).fetchall()
    short = {int(r["team_id"]): r["short_name"] for r in rows}
    name = {int(r["team_id"]): r["name"] for r in rows}
    return short, name


def _strength_lookup(conn: sqlite3.Connection, season: str, as_of_gw: int) -> dict[int, dict[str, float]]:
    """Team attack / defence proxies from as-of features (mean across squad)."""
    rows = conn.execute(
        """
        SELECT team_id, features_json
        FROM feat_player_asof
        WHERE season = ? AND as_of_gw = ? AND team_id IS NOT NULL
        """,
        (season, as_of_gw),
    ).fetchall()
    acc: dict[int, list[tuple[float, float]]] = {}
    for r in rows:
        feats = json.loads(r["features_json"]) if r["features_json"] else {}
        att = feats.get("team_attack")
        deff = feats.get("team_defence")
        if att is None and deff is None:
            continue
        try:
            acc.setdefault(int(r["team_id"]), []).append((float(att or 0), float(deff or 0)))
        except (TypeError, ValueError):
            continue
    out: dict[int, dict[str, float]] = {}
    for tid, vals in acc.items():
        out[tid] = {
            "attack": sum(v[0] for v in vals) / len(vals),
            "defence": sum(v[1] for v in vals) / len(vals),
        }
    return out


def player_fixtures(
    conn: sqlite3.Connection,
    season: str,
    as_of_gw: int,
    team_id: int | None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    if team_id is None:
        return []
    settings = settings or load_settings()
    ticker = fixture_ticker(conn, season, as_of_gw, settings)
    for team in ticker.get("teams", []):
        if team["team_id"] == team_id:
            return team["fixtures"]
    return []


def fixture_ticker(
    conn: sqlite3.Connection,
    season: str,
    as_of_gw: int,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings()
    short, names = _team_maps(conn, season)
    strength = _strength_lookup(conn, season, as_of_gw)
    fx = conn.execute(
        """
        SELECT fixture_id, event, team_h, team_a, finished, team_h_difficulty, team_a_difficulty
        FROM fact_fixture WHERE season = ?
        """,
        (season,),
    ).fetchall()
    fx_df = pd.DataFrame([dict(r) for r in fx])
    if fx_df.empty:
        return {"season": season, "as_of_gw": as_of_gw, "events": [], "teams": []}
    upc = upcoming_fixtures(fx_df, as_of_gw, settings.horizon_gws)
    events = sorted({int(e) for e in upc["event"].dropna().unique()})

    # Mean Fixtures (O) score by team — explains crowded picks
    # Prefer players likely in the model pool (priced + some minutes signal via M)
    o_rows = conn.execute(
        """
        SELECT s.team_id AS team_id, AVG(s.O) AS avg_o, AVG(s.overall) AS avg_overall, COUNT(*) AS n
        FROM score_player_asof s
        LEFT JOIN feat_player_asof f
          ON f.season = s.season AND f.as_of_gw = s.as_of_gw AND f.fpl_id = s.fpl_id
        WHERE s.season = ? AND s.as_of_gw = ? AND s.team_id IS NOT NULL
          AND COALESCE(f.price, 0) >= 4.0 AND s.M >= 40
        GROUP BY s.team_id
        """,
        (season, as_of_gw),
    ).fetchall()
    o_by_team = {int(r["team_id"]): r for r in o_rows if int(r["n"]) >= 3}

    mean_att = (
        sum(v["attack"] for v in strength.values()) / len(strength) if strength else 1.2
    )
    mean_def = (
        sum(v["defence"] for v in strength.values()) / len(strength) if strength else -1.2
    )

    teams_out = []
    for tid, abbr in sorted(short.items(), key=lambda x: x[1] or ""):
        fxs = upc.loc[(upc["team_h"] == tid) | (upc["team_a"] == tid)]
        fixtures = []
        att_ease_vals = []
        def_ease_vals = []
        for _, fr in fxs.sort_values("event").iterrows():
            home = int(fr["team_h"]) == tid
            opp = int(fr["team_a"]) if home else int(fr["team_h"])
            opp_att = strength.get(opp, {}).get("attack", mean_att)
            opp_def = strength.get(opp, {}).get("defence", mean_def)
            attack_ease = float(-opp_def)  # leaky defence = good for attackers
            def_ease = float(-opp_att)  # weak attack = good for CS
            att_ease_vals.append(attack_ease)
            def_ease_vals.append(def_ease)
            fixtures.append(
                {
                    "event": int(fr["event"]),
                    "opp": short.get(opp, str(opp)),
                    "opp_name": names.get(opp, ""),
                    "home": home,
                    "label": f"{short.get(opp, opp)} ({'H' if home else 'A'})",
                    "attack_ease": round(attack_ease, 2),
                    "def_ease": round(def_ease, 2),
                    "blank": False,
                }
            )
        # Fill blanks for missing events in horizon
        have = {f["event"] for f in fixtures}
        for ev in events:
            if ev not in have:
                fixtures.append(
                    {
                        "event": ev,
                        "opp": "-",
                        "opp_name": "Blank",
                        "home": None,
                        "label": "BLNK",
                        "attack_ease": 0.0,
                        "def_ease": 0.0,
                        "blank": True,
                    }
                )
        fixtures.sort(key=lambda f: f["event"])
        oinfo = o_by_team.get(tid)
        teams_out.append(
            {
                "team_id": tid,
                "team": abbr,
                "team_name": names.get(tid, abbr),
                "avg_fixtures": _round(oinfo["avg_o"], 1) if oinfo else None,
                "avg_overall": _round(oinfo["avg_overall"], 1) if oinfo else None,
                "attack_ease_avg": round(sum(att_ease_vals) / len(att_ease_vals), 2) if att_ease_vals else None,
                "def_ease_avg": round(sum(def_ease_vals) / len(def_ease_vals), 2) if def_ease_vals else None,
                "fixtures": fixtures,
            }
        )
    teams_out.sort(key=lambda t: (-(t["avg_fixtures"] or 0), t["team"] or ""))
    return {
        "season": season,
        "as_of_gw": as_of_gw,
        "horizon_gws": settings.horizon_gws,
        "events": events,
        "teams": teams_out,
    }


def build_xi(
    conn: sqlite3.Connection,
    season: str,
    as_of_gw: int,
    *,
    budget: float = 100.0,
    objective: str = "xpts",
) -> dict[str, Any]:
    from fpl_model.squad.optimize import build_squad

    players = list_players(
        conn,
        season,
        as_of_gw,
        sort=objective if objective in ("xpts", "overall") else "overall",
        limit=2000,
    )
    result = build_squad(players, budget=budget, objective=objective)

    def slim(p: dict[str, Any]) -> dict[str, Any]:
        return {
            "fpl_id": p["fpl_id"],
            "web_name": p["web_name"],
            "position": p["position"],
            "team": p.get("team"),
            "team_id": p.get("team_id"),
            "price": p.get("price"),
            "xpts": p.get("xpts"),
            "overall": p.get("overall"),
            "upside": p.get("upside"),
            "action": p.get("action"),
        }

    return {
        "season": season,
        "as_of_gw": as_of_gw,
        "ok": result["ok"],
        "budget": result["budget"],
        "spent": result["spent"],
        "remaining": result["remaining"],
        "objective": result["objective"],
        "squad_objective": result["squad_objective"],
        "xi_objective": result["xi_objective"],
        "formation": result["formation"],
        "captain": slim(result["captain"]) if result.get("captain") else None,
        "xi": [slim(p) for p in result["xi"]],
        "bench": [slim(p) for p in result["bench"]],
        "squad": [slim(p) for p in result["squad"]],
        "by_position": {k: [slim(p) for p in v] for k, v in result["by_position"].items()},
    }


def transfers_for_squad(
    conn: sqlite3.Connection,
    season: str,
    as_of_gw: int,
    squad_ids: list[int],
    *,
    bank: float = 0.0,
    free_transfers: int = 1,
    objective: str = "xpts",
) -> dict[str, Any]:
    pool = list_players(conn, season, as_of_gw, sort=objective, limit=2000)
    by_id = {p["fpl_id"]: p for p in pool}
    squad = [by_id[i] for i in squad_ids if i in by_id]
    result = suggest_transfers(
        squad,
        pool,
        bank=bank,
        free_transfers=free_transfers,
        objective=objective,
    )
    result["season"] = season
    result["as_of_gw"] = as_of_gw
    result["squad"] = [
        {
            "fpl_id": p["fpl_id"],
            "web_name": p["web_name"],
            "position": p["position"],
            "team": p.get("team"),
            "price": p.get("price"),
            "xpts": p.get("xpts"),
            "overall": p.get("overall"),
            "action": p.get("action"),
        }
        for p in squad
    ]
    return result


def _round(v: Any, nd: int = 1) -> float | None:
    if v is None:
        return None
    try:
        return round(float(v), nd)
    except (TypeError, ValueError):
        return None


def _player_row(r: sqlite3.Row) -> dict[str, Any]:
    return {
        "fpl_id": int(r["fpl_id"]),
        "web_name": r["web_name"],
        "position": r["position"],
        "team_id": r["team_id"],
        "team": r["team"],
        "team_name": r["team_name"],
        "status": r["status"] if "status" in r.keys() else None,
        "selected_by_percent": _round(r["selected_by_percent"], 1) if "selected_by_percent" in r.keys() else None,
        "price": _round(r["price"], 1) if "price" in r.keys() else None,
        "overall": _round(r["overall"]),
        "upside": _round(r["upside"]),
        "xpts": _round(r["xpts"], 2),
        "action": r["action"],
        "Q": _round(r["Q"]),
        "U": _round(r["U"]),
        "F": _round(r["F"]),
        "R": _round(r["R"]),
        "T": _round(r["T"]),
        "M": _round(r["M"]),
        "O": _round(r["O"]),
        "V": _round(r["V"]),
    }
