from __future__ import annotations

import json
import math
import sqlite3
from typing import Any

import pandas as pd

from fpl_model.config import Settings, load_settings
from fpl_model.features.asof import upcoming_fixtures
from fpl_model.scoring.clean_sheet import (
    ATTACK_EASE_EASY,
    ATTACK_EASE_HARD,
    ATTACK_FDR_CUTS,
    PCS_EASY,
    PCS_FDR_CUTS,
    PCS_HARD,
    attack_fixture_ease,
    ease_band,
    fdr_from_cuts,
    pack_match_for_team,
)
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
    sort: str = "xpts",
    limit: int = 400,
    ids: list[int] | None = None,
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
    order = sort_map.get(sort, "s.xpts")
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
    if ids:
        id_list = [int(i) for i in ids]
        where.append(f"s.fpl_id IN ({','.join('?' * len(id_list))})")
        params.extend(id_list)
        limit = max(int(limit), len(id_list))
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
    """Venue-split team xG from as-of features (same rates the scorer packed)."""
    rows = conn.execute(
        """
        SELECT team_id, features_json
        FROM feat_player_asof
        WHERE season = ? AND as_of_gw = ? AND team_id IS NOT NULL
        """,
        (season, as_of_gw),
    ).fetchall()
    keys = (
        "team_attack",
        "team_defence",
        "team_attack_home",
        "team_attack_away",
        "team_defence_home",
        "team_defence_away",
        "team_tilt",
    )
    acc: dict[int, list[dict[str, float]]] = {}
    for r in rows:
        feats = json.loads(r["features_json"]) if r["features_json"] else {}
        parsed: dict[str, float] = {}
        for k in keys:
            v = feats.get(k)
            if v is None:
                continue
            try:
                parsed[k] = float(v)
            except (TypeError, ValueError):
                continue
        if not parsed:
            continue
        att = parsed.get("team_attack")
        deff = parsed.get("team_defence")
        if att is not None:
            parsed.setdefault("team_attack_home", att)
            parsed.setdefault("team_attack_away", att)
        if deff is not None:
            parsed.setdefault("team_defence_home", deff)
            parsed.setdefault("team_defence_away", deff)
        acc.setdefault(int(r["team_id"]), []).append(parsed)
    out: dict[int, dict[str, float]] = {}
    for tid, vals in acc.items():
        merged: dict[str, float] = {}
        for k in keys:
            nums = [v[k] for v in vals if k in v]
            if nums:
                merged[k] = sum(nums) / len(nums)
        out[tid] = merged
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

    atk_vals = [v["team_attack"] for v in strength.values() if "team_attack" in v]
    league_xg = float(sum(atk_vals) / len(atk_vals)) if atk_vals else 1.35
    if not math.isfinite(league_xg) or league_xg <= 0.4:
        league_xg = 1.35

    teams_out = []
    for tid, abbr in sorted(short.items(), key=lambda x: x[1] or ""):
        fxs = upc.loc[(upc["team_h"] == tid) | (upc["team_a"] == tid)]
        fixtures = []
        att_ease_vals = []
        def_ease_vals = []
        for _, fr in fxs.sort_values("event").iterrows():
            home = int(fr["team_h"]) == tid
            opp = int(fr["team_a"]) if home else int(fr["team_h"])
            packed = pack_match_for_team(
                home=home,
                team_id=tid,
                opp_id=opp,
                lookup=strength,
                league_xg=league_xg,
                position="MID",
            )
            leak = float(packed["opp_defence_leak"])
            own_xg = float(packed["own_xg"])
            pcs = float(packed["p_cs"])
            att_ease = attack_fixture_ease(leak, own_xg, league_xg)
            att_ease_vals.append(att_ease)
            def_ease_vals.append(pcs)
            ha = "H" if home else "A"
            fixtures.append(
                {
                    "event": int(fr["event"]),
                    "opp": short.get(opp, str(opp)),
                    "opp_name": names.get(opp, ""),
                    "home": home,
                    "label": f"{short.get(opp, opp)} ({ha})",
                    "attack_ease": round(att_ease, 2),
                    "def_ease": round(pcs, 3),
                    "leak": round(leak, 2),
                    "own_xg": round(own_xg, 2),
                    "p_cs": round(pcs, 3),
                    "opp_xg": round(float(packed["opp_attack"]), 2),
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
                        "leak": None,
                        "own_xg": None,
                        "p_cs": None,
                        "opp_xg": None,
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
                "def_ease_avg": round(sum(def_ease_vals) / len(def_ease_vals), 3) if def_ease_vals else None,
                "fixtures": fixtures,
            }
        )

    for t in teams_out:
        for f in t["fixtures"]:
            if f["blank"]:
                f["attack_band"] = "blank"
                f["def_band"] = "blank"
                f["rating"] = None
                f["title"] = "Blank"
                continue
            ease = float(f["attack_ease"])
            leak = float(f["leak"])
            pcs = float(f["p_cs"])
            f["attack_band"] = ease_band(ease, ATTACK_EASE_HARD, ATTACK_EASE_EASY, higher_is_easier=True)
            f["def_band"] = ease_band(pcs, PCS_HARD, PCS_EASY, higher_is_easier=True)
            f["rating"] = fdr_from_cuts(ease, ATTACK_FDR_CUTS, higher_is_easier=True)
            f["def_rating"] = fdr_from_cuts(pcs, PCS_FDR_CUTS, higher_is_easier=True)
            f["title"] = (
                f"{f['opp_name'] or f['opp']} ({'H' if f['home'] else 'A'}) · "
                f"ease {ease:.2f} (leak {leak:.2f} × own xG {f['own_xg']:.2f}) · "
                f"P(CS) {pcs:.0%} · {f['rating']}"
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


def _gw_fixture_labels(
    conn: sqlite3.Connection, season: str, as_of_gw: int, team_ids: list[int]
) -> dict[int, str]:
    wanted = {int(t) for t in team_ids if t is not None}
    if not wanted:
        return {}
    short = {
        int(r["team_id"]): r["short_name"]
        for r in conn.execute("SELECT team_id, short_name FROM dim_team WHERE season = ?", (season,))
    }
    rows = conn.execute(
        """
        SELECT team_h, team_a FROM fact_fixture
        WHERE season = ? AND event = ? AND team_h IS NOT NULL AND team_a IS NOT NULL
        """,
        (season, as_of_gw),
    ).fetchall()
    labels: dict[int, list[str]] = {tid: [] for tid in wanted}
    for r in rows:
        h, a = int(r["team_h"]), int(r["team_a"])
        if h in labels:
            labels[h].append(f"{short.get(a, a)} (H)")
        if a in labels:
            labels[a].append(f"{short.get(h, h)} (A)")
    return {tid: " / ".join(v) if v else "BLANK" for tid, v in labels.items()}


def lineup_for_squad(
    conn: sqlite3.Connection,
    season: str,
    as_of_gw: int,
    squad_ids: list[int],
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Best legal starting XI from a saved 15, using next-GW xPts only."""
    from dataclasses import replace

    from fpl_model.config import load_weights
    from fpl_model.scoring.clean_sheet import isolate_event
    from fpl_model.scoring.engine import load_feats
    from fpl_model.scoring.xpts import compute_xpts
    from fpl_model.squad.optimize import pick_xi

    settings = settings or load_settings()
    ids = [int(i) for i in squad_ids]
    empty: dict[str, Any] = {
        "ok": False,
        "season": season,
        "as_of_gw": as_of_gw,
        "event": as_of_gw,
        "xi": [],
        "bench": [],
        "captain": None,
        "formation": None,
        "xi_objective": 0.0,
    }
    if len(ids) != 15 or len(set(ids)) != 15:
        empty["error"] = f"need a full 15-man squad (currently {len(set(ids))})"
        return empty

    feats = load_feats(conn, season, as_of_gw, fpl_ids=ids)
    if feats.empty or "fpl_id" not in feats.columns:
        empty["error"] = "no features for this as-of"
        return empty
    fid_col = feats["fpl_id"]
    if isinstance(fid_col, pd.DataFrame):
        feats = feats.copy()
        feats["fpl_id"] = fid_col.iloc[:, 0]
    feats = feats.drop_duplicates(subset=["fpl_id"], keep="last")
    if len(feats) != 15:
        empty["error"] = "scores missing for some squad players"
        return empty

    gw1 = replace(settings, horizon_gws=1)
    if "fixtures_json" in feats.columns:
        feats["fixtures_json"] = feats["fixtures_json"].map(lambda j: isolate_event(j, as_of_gw))
    feats["xpts_gw"] = compute_xpts(feats, load_weights(), gw1)

    by_id = {p["fpl_id"]: p for p in list_players(conn, season, as_of_gw, sort="xpts", ids=ids)}
    xmap = pd.to_numeric(feats.set_index("fpl_id")["xpts_gw"], errors="coerce")
    team_ids = [p.get("team_id") for p in by_id.values() if p.get("team_id") is not None]
    fx_by_team = _gw_fixture_labels(conn, season, as_of_gw, team_ids)

    squad: list[dict[str, Any]] = []
    for fid in ids:
        p = by_id.get(fid)
        if p is None:
            empty["error"] = "scores missing for some squad players"
            return empty
        row = dict(p)
        try:
            row["xpts_gw"] = round(float(xmap.loc[fid]), 2)
        except (KeyError, TypeError, ValueError):
            row["xpts_gw"] = 0.0
        tid = row.get("team_id")
        row["fixture"] = fx_by_team.get(int(tid), "BLANK") if tid is not None else "BLANK"
        squad.append(row)

    pack = pick_xi(squad, objective="xpts_gw")

    def slim(p: dict[str, Any]) -> dict[str, Any]:
        return {
            "fpl_id": p["fpl_id"],
            "web_name": p["web_name"],
            "position": p["position"],
            "team": p.get("team"),
            "team_id": p.get("team_id"),
            "price": p.get("price"),
            "xpts": p.get("xpts"),
            "xpts_gw": round(float(p.get("xpts_gw") or 0), 2),
            "overall": p.get("overall"),
            "fixture": p.get("fixture"),
            "action": p.get("action"),
        }

    cap = pack.get("captain")
    return {
        "ok": bool(pack.get("xi")),
        "season": season,
        "as_of_gw": as_of_gw,
        "event": as_of_gw,
        "horizon_gws": 1,
        "objective": "xpts_gw",
        "formation": pack["formation"],
        "xi_objective": pack["xi_objective"],
        "captain": slim(cap) if cap else None,
        "xi": [slim(p) for p in pack["xi"]],
        "bench": [slim(p) for p in pack["bench"]],
        "squad": [slim(p) for p in squad],
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
