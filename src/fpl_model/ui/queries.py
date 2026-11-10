from __future__ import annotations

import json
import math
import re
import sqlite3
from typing import Any

import pandas as pd

from fpl_model.config import Settings, load_settings
from fpl_model.features.asof import upcoming_fixtures
from fpl_model.scoring.availability import minutes_risk
from fpl_model.scoring.clean_sheet import (
    ATTACK_EASE_EASY,
    ATTACK_EASE_HARD,
    ATTACK_FDR_CUTS,
    PCS_EASY,
    PCS_FDR_CUTS,
    PCS_HARD,
    VENUE_THIN_N,
    attack_fixture_ease,
    ease_band,
    fdr_from_cuts,
    pack_match_for_team,
)
from fpl_model.squad.transfers import annotate_fixture_flags, rebuild_on_chip, suggest_transfers
from fpl_model.squad.wildcard import MAX_WC_HORIZON, PERSIST, advise_wildcard, clip_windows


def available_snapshots(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT s.season, s.as_of_gw, COUNT(*) AS n,
               COALESCE(m.is_beta, 0) AS is_beta
        FROM score_player_asof s
        LEFT JOIN dim_asof_snapshot m
          ON m.season = s.season AND m.as_of_gw = s.as_of_gw
        GROUP BY s.season, s.as_of_gw, m.is_beta
        ORDER BY s.season DESC, s.as_of_gw DESC
        """
    ).fetchall()
    return [
        {
            "season": r["season"],
            "as_of_gw": int(r["as_of_gw"]),
            "n": int(r["n"]),
            "is_beta": bool(r["is_beta"]),
        }
        for r in rows
    ]


def default_snapshot(conn: sqlite3.Connection, preferred_season: str | None = None) -> dict[str, Any] | None:
    snaps = available_snapshots(conn)
    if not snaps:
        return None
    if preferred_season:
        for s in snaps:
            if s["season"] == preferred_season:
                return s
    return snaps[0]


def list_clubs(conn: sqlite3.Connection, season: str | None = None) -> dict[str, list[dict[str, Any]]]:
    where = ""
    params: list[Any] = []
    if season:
        where = "WHERE season = ?"
        params.append(season)
    rows = conn.execute(
        f"""
        SELECT season, team_id, short_name, name
        FROM dim_team
        {where}
        ORDER BY season, short_name
        """,
        params,
    ).fetchall()
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(r["season"], []).append(
            {
                "team_id": int(r["team_id"]),
                "short_name": r["short_name"],
                "name": r["name"],
            }
        )
    return out


def list_players(
    conn: sqlite3.Connection,
    season: str,
    as_of_gw: int,
    *,
    position: str | None = None,
    q: str | None = None,
    action: str | None = None,
    team_id: int | None = None,
    sort: str = "xpts",
    limit: int = 400,
    ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    sort_map = {
        "overall": "s.overall",
        "upside": "s.upside",
        "xpts": "s.xpts",
        "xpts_next": "CAST(json_extract(s.card_json, '$.xpts_next') AS REAL)",
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
    if team_id is not None:
        where.append("s.team_id = ?")
        params.append(int(team_id))
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
            s.Q, s.U, s.F, s.R, s.T, s.M, s.O, s.V,
            json_extract(s.card_json, '$.xpts_next') AS xpts_next,
            json_extract(s.card_json, '$.xpts_gws') AS xpts_gws,
            json_extract(s.card_json, '$.flags') AS flags,
            json_extract(s.card_json, '$.minutes_risk') AS minutes_risk,
            json_extract(s.card_json, '$.availability') AS availability,
            p.news, p.chance_of_playing_this, p.chance_of_playing_next
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
    cal = _calibration_for_nudge(conn, season, as_of_gw)
    out = [_apply_cal_nudge(_player_row(r), cal) for r in rows]
    lw = last_week_map(conn, season, as_of_gw, [p["fpl_id"] for p in out])
    for p in out:
        p["last_week"] = lw.get(int(p["fpl_id"]))
    return out


def get_player(conn: sqlite3.Connection, season: str, as_of_gw: int, fpl_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT
            s.fpl_id, s.web_name, s.position, s.team_id, s.card_json,
            t.short_name AS team, t.name AS team_name,
            p.status, p.selected_by_percent, p.first_name, p.second_name,
            p.news, p.chance_of_playing_this, p.chance_of_playing_next,
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
    if out.get("xpts_next") is None and card.get("xpts_next") is not None:
        out["xpts_next"] = _round(card.get("xpts_next"), 2)
    if out.get("xpts_gws") is None:
        out["xpts_gws"] = _parse_xpts_gws(card.get("xpts_gws"))
    if not out.get("minutes_risk") and isinstance(card.get("minutes_risk"), dict):
        out["minutes_risk"] = card.get("minutes_risk")
    if out.get("availability") is None and card.get("availability") is not None:
        out["availability"] = _round(card.get("availability"), 3)
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
    out["news"] = row["news"] if "news" in row.keys() else None
    out["chance_of_playing_this"] = (
        _round(row["chance_of_playing_this"], 0) if "chance_of_playing_this" in row.keys() else None
    )
    out["chance_of_playing_next"] = (
        _round(row["chance_of_playing_next"], 0) if "chance_of_playing_next" in row.keys() else None
    )
    if not out.get("minutes_risk"):
        out["minutes_risk"] = minutes_risk(
            availability=card.get("availability"),
            status=out.get("status"),
            news=out.get("news"),
            chance_this=row["chance_of_playing_this"] if "chance_of_playing_this" in row.keys() else None,
            chance_next=row["chance_of_playing_next"] if "chance_of_playing_next" in row.keys() else None,
            position=out.get("position"),
            minutes=feats.get("minutes"),
            starts=feats.get("starts"),
            appearances=feats.get("appearances"),
            slot_security=feats.get("slot_security"),
            pos_slot_rank=feats.get("pos_slot_rank"),
            start_pct_l5=feats.get("start_pct_l5"),
        )
    cal = _calibration_for_nudge(conn, season, as_of_gw)
    out = _apply_cal_nudge(out, cal)
    out["last_week"] = last_week_outcome(conn, season, as_of_gw, int(row["fpl_id"]))
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
        "psxg": num("psxg", 2),
        "psxg_net": num("psxg_net", 2),
        "psxg_net90": num("psxg_net90", 2),
        "shot_stopping": num("shot_stopping", 2),
    }


def _team_maps(conn: sqlite3.Connection, season: str) -> tuple[dict[int, str], dict[int, str]]:
    rows = conn.execute(
        "SELECT team_id, short_name, name FROM dim_team WHERE season = ?",
        (season,),
    ).fetchall()
    short = {int(r["team_id"]): r["short_name"] for r in rows}
    name = {int(r["team_id"]): r["name"] for r in rows}
    return short, name


def _venue_sample_n(conn: sqlite3.Connection, season: str, as_of_gw: int) -> dict[int, dict[str, int]]:
    rows = conn.execute(
        """
        SELECT team_h, team_a FROM fact_fixture
        WHERE season = ? AND event < ? AND finished = 1
          AND team_h IS NOT NULL AND team_a IS NOT NULL
        """,
        (season, int(as_of_gw)),
    ).fetchall()
    out: dict[int, dict[str, int]] = {}
    for r in rows:
        h, a = int(r["team_h"]), int(r["team_a"])
        out.setdefault(h, {"home": 0, "away": 0})
        out.setdefault(a, {"home": 0, "away": 0})
        out[h]["home"] += 1
        out[a]["away"] += 1
    return out


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
        "venue_n_home",
        "venue_n_away",
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
    venue_n = _venue_sample_n(conn, season, as_of_gw)
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
            sample = venue_n.get(int(tid), {"home": 0, "away": 0})
            n_venue = int(sample["home"] if home else sample["away"])
            if n_venue <= 0:
                feat_n = strength.get(int(tid), {}).get("venue_n_home" if home else "venue_n_away")
                if feat_n is not None:
                    try:
                        n_venue = int(float(feat_n))
                    except (TypeError, ValueError):
                        n_venue = 0
            thin = n_venue < VENUE_THIN_N
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
                    "venue_n": n_venue,
                    "thin_venue": thin,
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
                        "venue_n": None,
                        "thin_venue": False,
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
                + (f" · n={int(f['venue_n'])} {('home' if f['home'] else 'away')} (thin venue)" if f.get("thin_venue") else "")
            )

    teams_out.sort(key=lambda t: (-(t["avg_fixtures"] or 0), t["team"] or ""))
    return {
        "season": season,
        "as_of_gw": as_of_gw,
        "horizon_gws": settings.horizon_gws,
        "events": events,
        "teams": teams_out,
    }


def attach_weekly_xpts(
    conn: sqlite3.Connection,
    season: str,
    as_of_gw: int,
    players: list[dict[str, Any]],
    settings: Settings | None = None,
) -> None:
    """Per-GW xPts over the scoring window (blanks 0, DGWs both matches)."""
    from dataclasses import replace

    from fpl_model.config import load_weights
    from fpl_model.scoring.clean_sheet import isolate_event
    from fpl_model.scoring.engine import load_feats
    from fpl_model.scoring.xpts import compute_xpts

    if not players:
        return
    settings = settings or load_settings()
    hz = max(1, int(settings.horizon_gws))
    missing = [
        p
        for p in players
        if not (isinstance(p.get("xpts_gws"), list) and len(p["xpts_gws"]) >= hz)
    ]
    if not missing:
        return
    feats = load_feats(conn, season, as_of_gw, fpl_ids=[int(p["fpl_id"]) for p in missing])
    if feats.empty or "fpl_id" not in feats.columns:
        for p in missing:
            p["xpts_gws"] = [0.0] * hz
        return
    gw1 = replace(settings, horizon_gws=1)
    weights = load_weights()
    by_id: dict[int, list[float]] = {int(p["fpl_id"]): [0.0] * hz for p in missing}
    base_fx = feats["fixtures_json"] if "fixtures_json" in feats.columns else None
    for offset in range(hz):
        event = int(as_of_gw) + offset
        df = feats
        if base_fx is not None:
            df = feats.copy()
            df["fixtures_json"] = base_fx.map(lambda j, ev=event: isolate_event(j, ev))
        if offset == 0 and "availability" in df.columns:
            df["availability_horizon"] = df["availability"]
        elif offset > 0 and "availability_later" in df.columns:
            df["availability_horizon"] = df["availability_later"]
        series = compute_xpts(df, weights, gw1)
        for fid, val in zip(df["fpl_id"].tolist(), series.tolist()):
            fid = int(fid)
            if fid in by_id:
                by_id[fid][offset] = round(float(val or 0), 3)
    for p in missing:
        p["xpts_gws"] = by_id.get(int(p["fpl_id"]), [0.0] * hz)


def build_xi(
    conn: sqlite3.Connection,
    season: str,
    as_of_gw: int,
    *,
    budget: float = 100.0,
    objective: str = "xpts",
    exclude: list[int] | None = None,
    exclude_teams: list[int] | None = None,
    include: list[int] | None = None,
    lock: list[int] | None = None,
) -> dict[str, Any]:
    from fpl_model.squad.optimize import build_squad

    players = list_players(
        conn,
        season,
        as_of_gw,
        sort=objective if objective in ("xpts", "overall") else "overall",
        limit=2000,
    )
    settings = load_settings()
    attach_weekly_xpts(conn, season, as_of_gw, players, settings)
    ban = [int(x) for x in (exclude or [])]
    team_ban = [int(x) for x in (exclude_teams or [])]
    allow = [int(x) for x in (include or [])]
    locked = [int(x) for x in (lock or [])]
    result = build_squad(
        players,
        budget=budget,
        objective=objective,
        exclude=ban,
        exclude_teams=team_ban,
        include=allow,
        lock=locked,
    )

    def slim(p: dict[str, Any]) -> dict[str, Any]:
        return {
            "fpl_id": p["fpl_id"],
            "web_name": p["web_name"],
            "position": p["position"],
            "team": p.get("team"),
            "team_id": p.get("team_id"),
            "price": p.get("price"),
            "xpts": p.get("xpts"),
            "xpts_next": p.get("xpts_next"),
            "xpts_gws": p.get("xpts_gws"),
            "overall": p.get("overall"),
            "upside": p.get("upside"),
            "action": p.get("action"),
            "combined": p.get("combined"),
            "bench_only": p.get("_xi_ok") is False,
            "locked": int(p["fpl_id"]) in set(locked),
        }

    excluded_players = [slim(p) for p in players if int(p["fpl_id"]) in set(ban)]
    included_players = [slim(p) for p in players if int(p["fpl_id"]) in set(allow)]
    locked_players = [slim(p) for p in players if int(p["fpl_id"]) in set(locked)]
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
        "xi_xpts": round(sum(float(p.get("xpts") or 0) for p in result["xi"]), 2),
        "xi_overall": round(sum(float(p.get("overall") or 0) for p in result["xi"]), 1),
        "formation": result["formation"],
        "captain": slim(result["captain"]) if result.get("captain") else None,
        "xi": [slim(p) for p in result["xi"]],
        "bench": [slim(p) for p in result["bench"]],
        "squad": [slim(p) for p in result["squad"]],
        "by_position": {k: [slim(p) for p in v] for k, v in result["by_position"].items()},
        "excluded": excluded_players,
        "excluded_ids": sorted(set(ban)),
        "excluded_teams": sorted(set(team_ban)),
        "included": included_players,
        "included_ids": sorted(set(allow)),
        "locked": locked_players,
        "locked_ids": sorted(set(locked)),
        "error": result.get("error"),
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


def _lineup_event(as_of_gw: int, event: int | None, horizon_gws: int) -> int:
    hz = max(1, int(horizon_gws))
    lo, hi = int(as_of_gw), int(as_of_gw) + hz - 1
    if event is None:
        return lo
    return max(lo, min(int(event), hi))


def lineup_for_squad(
    conn: sqlite3.Connection,
    season: str,
    as_of_gw: int,
    squad_ids: list[int],
    *,
    event: int | None = None,
    chip: str | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Best legal starting XI from a saved 15, using one GW's fixtures (T, T+1, …)."""
    from dataclasses import replace

    from fpl_model.config import load_weights
    from fpl_model.scoring.clean_sheet import isolate_event
    from fpl_model.scoring.engine import load_feats
    from fpl_model.scoring.xpts import compute_xpts
    from fpl_model.squad.optimize import pick_xi

    settings = settings or load_settings()
    target = _lineup_event(as_of_gw, event, settings.horizon_gws)
    offset = target - int(as_of_gw)
    ids = [int(i) for i in squad_ids]
    empty: dict[str, Any] = {
        "ok": False,
        "season": season,
        "as_of_gw": as_of_gw,
        "event": target,
        "offset": offset,
        "xi": [],
        "bench": [],
        "captain": None,
        "formation": None,
        "xi_objective": 0.0,
    }
    if len(ids) != 15 or len(set(ids)) != 15:
        empty["error"] = f"need a full 15-man squad (currently {len(set(ids))})"
        return empty

    by_id = {p["fpl_id"]: p for p in list_players(conn, season, as_of_gw, sort="xpts", ids=ids)}
    stored_ok = all(
        isinstance(p.get("xpts_gws"), list) and len(p["xpts_gws"]) > offset for p in by_id.values()
    )
    if stored_ok:
        xmap = {int(fid): float(p["xpts_gws"][offset] or 0) for fid, p in by_id.items()}
    else:
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
            feats["fixtures_json"] = feats["fixtures_json"].map(lambda j: isolate_event(j, target))
        if offset == 0 and "availability" in feats.columns:
            feats["availability_horizon"] = feats["availability"]
        elif offset > 0 and "availability_later" in feats.columns:
            feats["availability_horizon"] = feats["availability_later"]
        feats["xpts_gw"] = compute_xpts(feats, load_weights(), gw1)
        xmap = {int(k): float(v or 0) for k, v in pd.to_numeric(feats.set_index("fpl_id")["xpts_gw"], errors="coerce").items()}

    team_ids = [p.get("team_id") for p in by_id.values() if p.get("team_id") is not None]
    fx_by_team = _gw_fixture_labels(conn, season, target, team_ids)
    cal = _calibration_for_nudge(conn, season, as_of_gw)

    squad: list[dict[str, Any]] = []
    for fid in ids:
        p = by_id.get(fid)
        if p is None:
            empty["error"] = "scores missing for some squad players"
            return empty
        row = dict(p)
        raw_gw = float(xmap.get(int(fid), 0.0))
        nudge = calibration_nudge(cal, row.get("position"), raw_gw)
        row["xpts_gw_raw"] = round(raw_gw, 2)
        row["xpts_gw"] = round(raw_gw + nudge, 2)
        row["cal_nudge"] = nudge
        tid = row.get("team_id")
        row["fixture"] = fx_by_team.get(int(tid), "BLANK") if tid is not None else "BLANK"
        squad.append(row)

    pack = pick_xi(squad, objective="xpts_gw")
    from fpl_model.squad.chips import apply_chip

    chip_name = str(chip or "").lower()
    placed = None
    if chip_name in {"bboost", "3xc"}:
        placed = apply_chip(squad, chip_name, objective="xpts_gw")

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
            "availability": p.get("availability"),
            "minutes_risk": p.get("minutes_risk"),
            "selected_by_percent": p.get("selected_by_percent"),
        }

    cap = pack.get("captain")
    vice = pack.get("vice_captain")
    xi_objective = pack["xi_objective"]
    if placed and placed.get("ok"):
        xi_objective = placed["xi_objective"]
        cap = placed.get("captain") or cap
    pool = list_players(conn, season, as_of_gw, sort="xpts", limit=400)
    fh_pool = list_players(conn, season, as_of_gw, sort="xpts_next", limit=500)
    from fpl_model.squad.chips import freehit_indicator, scan_chip_weeks

    scan = scan_chip_weeks(squad, as_of_gw=as_of_gw)
    scan["freehit"] = freehit_indicator(squad, fh_pool, as_of_gw=as_of_gw)
    return {
        "ok": bool(pack.get("xi")),
        "season": season,
        "as_of_gw": as_of_gw,
        "event": target,
        "offset": offset,
        "horizon_gws": 1,
        "objective": "xpts_gw",
        "chip": chip_name or None,
        "include_bench": bool(placed and placed.get("include_bench")),
        "triple": bool(placed and placed.get("triple")),
        "chip_reason": (placed or {}).get("reason"),
        "formation": pack["formation"],
        "xi_objective": xi_objective,
        "captain": slim(cap) if cap else None,
        "vice_captain": slim(vice) if vice else None,
        "captain_ev": pack.get("captain_ev"),
        "vice_ev": pack.get("vice_ev"),
        "triple_ev": pack.get("triple_ev"),
        "captain_xpts": pack.get("captain_xpts"),
        "xi": [slim(p) for p in pack["xi"]],
        "bench": [slim(p) for p in pack["bench"]],
        "squad": [slim(p) for p in squad],
        "chip_scan": scan,
        "template": template_vs_squad(pool, ids),
    }


def _enrich_transfer_players(
    conn: sqlite3.Connection,
    season: str,
    as_of_gw: int,
    players: list[dict[str, Any]],
    horizon_gws: int,
) -> None:
    """Attach availability / xpts_next / fixtures_json for transfer suggestions."""
    if not players:
        return
    ids = [int(p["fpl_id"]) for p in players]
    placeholders = ",".join("?" * len(ids))
    feat_rows = conn.execute(
        f"""
        SELECT fpl_id, features_json FROM feat_player_asof
        WHERE season = ? AND as_of_gw = ? AND fpl_id IN ({placeholders})
        """,
        (season, as_of_gw, *ids),
    ).fetchall()
    score_rows = conn.execute(
        f"""
        SELECT fpl_id, card_json FROM score_player_asof
        WHERE season = ? AND as_of_gw = ? AND fpl_id IN ({placeholders})
        """,
        (season, as_of_gw, *ids),
    ).fetchall()
    feats: dict[int, dict] = {}
    for r in feat_rows:
        try:
            feats[int(r["fpl_id"])] = json.loads(r["features_json"] or "{}")
        except json.JSONDecodeError:
            feats[int(r["fpl_id"])] = {}
    cards: dict[int, dict] = {}
    for r in score_rows:
        try:
            cards[int(r["fpl_id"])] = json.loads(r["card_json"] or "{}")
        except json.JSONDecodeError:
            cards[int(r["fpl_id"])] = {}
    cal = _calibration_for_nudge(conn, season, as_of_gw)
    for p in players:
        fid = int(p["fpl_id"])
        f = feats.get(fid) or {}
        c = cards.get(fid) or {}
        p["availability"] = c.get("availability", f.get("availability", 1.0))
        p["availability_horizon"] = c.get("availability_horizon", f.get("availability_horizon", p["availability"]))
        p["xpts_next"] = c.get("xpts_next") if c.get("xpts_next") is not None else p.get("xpts_next")
        p["xpts_gws"] = c.get("xpts_gws") or p.get("xpts_gws")
        p["fixtures_json"] = f.get("fixtures_json", "{}")
        p["price_change"] = f.get("price_change", 0.0)
        p["minutes_risk"] = c.get("minutes_risk") or minutes_risk(
            availability=p.get("availability"),
            status=p.get("status"),
            news=p.get("news"),
            chance_this=f.get("chance_of_playing_this"),
            chance_next=f.get("chance_of_playing_next"),
            slot_security=f.get("slot_security"),
            pos_slot_rank=f.get("pos_slot_rank"),
            position=p.get("position"),
            start_pct_l5=f.get("start_pct_l5"),
            minutes=f.get("minutes"),
            starts=f.get("starts"),
            appearances=f.get("appearances"),
        )
        _apply_cal_nudge(p, cal)
    annotate_fixture_flags(players, horizon_gws)


def transfers_for_squad(
    conn: sqlite3.Connection,
    season: str,
    as_of_gw: int,
    squad_ids: list[int],
    *,
    bank: float = 0.0,
    budget: float = 100.0,
    free_transfers: int = 1,
    objective: str = "xpts",
    settings: Settings | None = None,
    exclude: list[int] | None = None,
    exclude_teams: list[int] | None = None,
    include: list[int] | None = None,
    keep: list[int] | None = None,
    want_transfers: int | None = None,
    committed: list | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings()
    pool = list_players(conn, season, as_of_gw, sort="xpts" if objective == "combined" else objective, limit=2000)
    _enrich_transfer_players(conn, season, as_of_gw, pool, settings.horizon_gws)
    by_id = {p["fpl_id"]: p for p in pool}
    squad = [by_id[i] for i in squad_ids if i in by_id]
    result = suggest_transfers(
        squad,
        pool,
        bank=bank,
        budget=budget,
        free_transfers=free_transfers,
        objective=objective,
        horizon_gws=settings.horizon_gws,
        exclude=exclude,
        exclude_teams=exclude_teams,
        include=include,
        keep=keep,
        want_transfers=want_transfers,
        committed=committed,
    )
    from fpl_model.squad.chips import freehit_indicator, scan_chip_weeks

    result["season"] = season
    result["as_of_gw"] = as_of_gw
    scan = scan_chip_weeks(squad, as_of_gw=as_of_gw)
    scan["freehit"] = freehit_indicator(squad, pool, as_of_gw=as_of_gw)
    result["chip_scan"] = scan
    result["squad"] = [
        {
            "fpl_id": p["fpl_id"],
            "web_name": p["web_name"],
            "position": p["position"],
            "team": p.get("team"),
            "price": p.get("price"),
            "xpts": p.get("xpts"),
            "xpts_next": p.get("xpts_next"),
            "overall": p.get("overall"),
            "action": p.get("action"),
            "availability": p.get("availability"),
            "blank_next": p.get("blank_next"),
            "minutes_risk": p.get("minutes_risk"),
        }
        for p in squad
    ]
    return result


def rebuild_for_squad(
    conn: sqlite3.Connection,
    season: str,
    as_of_gw: int,
    squad_ids: list[int],
    *,
    chip: str = "wildcard",
    bank: float = 0.0,
    budget: float = 100.0,
    objective: str = "xpts",
    settings: Settings | None = None,
    exclude: list[int] | None = None,
    exclude_teams: list[int] | None = None,
    include: list[int] | None = None,
    keep: list[int] | None = None,
    lock: list[int] | None = None,
    max_value: float | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings()
    pool = list_players(conn, season, as_of_gw, sort="xpts" if objective == "combined" else objective, limit=2000)
    _enrich_transfer_players(conn, season, as_of_gw, pool, settings.horizon_gws)
    by_id = {p["fpl_id"]: p for p in pool}
    squad = [by_id[i] for i in squad_ids if i in by_id]
    result = rebuild_on_chip(
        squad,
        pool,
        chip=chip,
        bank=bank,
        budget=budget,
        objective=objective,
        keep=keep,
        lock=lock,
        exclude=exclude,
        exclude_teams=exclude_teams,
        include=include,
        max_value=max_value,
    )
    result["season"] = season
    result["as_of_gw"] = as_of_gw

    def slim(p: dict[str, Any]) -> dict[str, Any]:
        return {
            "fpl_id": p["fpl_id"],
            "web_name": p["web_name"],
            "position": p["position"],
            "team": p.get("team"),
            "team_id": p.get("team_id"),
            "price": p.get("price"),
            "xpts": p.get("xpts"),
            "xpts_next": p.get("xpts_next"),
            "overall": p.get("overall"),
            "action": p.get("action"),
        }

    if result.get("squad"):
        result["squad"] = [slim(p) for p in result["squad"]]
        result["xi"] = [slim(p) for p in (result.get("xi") or [])]
        result["bench"] = [slim(p) for p in (result.get("bench") or [])]
        cap = result.get("captain")
        result["captain"] = slim(cap) if cap else None
        result["ids"] = [p["fpl_id"] for p in result["squad"]]
    return result


def wildcard_for_squad(
    conn: sqlite3.Connection,
    season: str,
    as_of_gw: int,
    squad_ids: list[int],
    *,
    bank: float = 0.0,
    budget: float = 100.0,
    settings: Settings | None = None,
    exclude: list[int] | None = None,
    exclude_teams: list[int] | None = None,
    include: list[int] | None = None,
    keep: list[int] | None = None,
    lock: list[int] | None = None,
    max_value: float | None = None,
    chips_used: list[str] | None = None,
    free_transfers: int = 1,
) -> dict[str, Any]:
    """Score current 15 vs wildcard 15s on 1/3/5/8/13/20/26/rest-of-season look-aheads."""
    from dataclasses import replace

    from fpl_model.config import load_weights
    from fpl_model.features.asof import attach_horizon_fixtures
    from fpl_model.scoring.engine import load_feats
    from fpl_model.scoring.xpts import compute_xpts_weeks

    settings = settings or load_settings()
    if len(squad_ids) != 15:
        return {"ok": False, "error": f"need a full 15-man squad (currently {len(set(squad_ids))})", "squad": []}
    fx = pd.read_sql_query(
        "SELECT event, team_h, team_a FROM fact_fixture WHERE season = ? AND event IS NOT NULL",
        conn,
        params=(season,),
    )
    if fx.empty:
        return {"ok": False, "error": "no fixtures in the DB", "squad": []}
    max_ev = int(pd.to_numeric(fx["event"], errors="coerce").max())
    remaining = max(1, max_ev - int(as_of_gw) + 1)
    cap = min(MAX_WC_HORIZON, remaining)
    windows = clip_windows(cap)
    feats = load_feats(conn, season, as_of_gw)
    if feats.empty:
        return {"ok": False, "error": "no features for this as-of", "squad": []}
    feats = attach_horizon_fixtures(feats, fx, as_of_gw, cap)
    weeks = compute_xpts_weeks(feats, load_weights(), replace(settings, horizon_gws=cap))
    by_fid: dict[int, list[float]] = {}
    for idx, row in weeks.iterrows():
        fid = int(feats.loc[idx, "fpl_id"])
        by_fid[fid] = [round(float(v), 3) for v in row.tolist()]

    pool = list_players(conn, season, as_of_gw, sort="xpts", limit=2000)
    _enrich_transfer_players(conn, season, as_of_gw, pool, cap)
    for p in pool:
        xs = by_fid.get(int(p["fpl_id"]))
        if xs:
            p["xpts_gws"] = xs
            p["xpts_next"] = xs[0]
            p["xpts"] = float(sum(xs[: settings.horizon_gws]))
    by_id = {p["fpl_id"]: p for p in pool}
    squad = [by_id[i] for i in squad_ids if i in by_id]
    if len(squad) != 15:
        return {"ok": False, "error": "scores missing for some squad players", "squad": []}
    result = advise_wildcard(
        squad,
        pool,
        windows=windows,
        bank=bank,
        budget=budget,
        keep=keep,
        lock=lock,
        exclude=exclude,
        exclude_teams=exclude_teams,
        include=include,
        max_value=max_value,
        chips_used=chips_used,
        free_transfers=free_transfers,
        as_of_gw=as_of_gw,
    )
    result["season"] = season
    result["as_of_gw"] = as_of_gw
    result["note"] = (
        f"xPts here pack {cap} GWs of fixtures (stored cards stay on {settings.horizon_gws} GWs). "
        "From ~8 GW out this is fixture-shape, not a forecast — later weeks are persistence-weighted "
        f"({PERSIST:.2f}^week, floor 0.40). Play/wait uses the next 5 GWs; 8+ GW is core shape. "
        "The WC 15 front-weights 3 GWs and may keep up to two cheap fixture stints — later weeks are left to ordinary FTs, with no planned transfers."
    )
    result["constraints"] = {
        "exclude": list(exclude or []),
        "exclude_teams": list(exclude_teams or []),
        "include": list(include or []),
        "keep": list(keep or []),
        "lock": list(lock or []),
        "max_value": None if max_value is None else round(float(max_value), 1),
    }
    return result


def latest_calibration(conn: sqlite3.Connection, season: str, as_of_gw: int | None = None) -> dict[str, Any] | None:
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "fact_calibrate_asof" not in tables:
        return None
    if as_of_gw is not None:
        row = conn.execute(
            """
            SELECT season, as_of_gw, n, mae, bias, unmatched, breakdown_json, computed_at
            FROM fact_calibrate_asof WHERE season = ? AND as_of_gw = ?
            """,
            (season, int(as_of_gw)),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT season, as_of_gw, n, mae, bias, unmatched, breakdown_json, computed_at
            FROM fact_calibrate_asof WHERE season = ? ORDER BY as_of_gw DESC LIMIT 1
            """,
            (season,),
        ).fetchone()
    if row is None:
        return None
    breakdown = {}
    try:
        breakdown = json.loads(row["breakdown_json"] or "{}")
    except json.JSONDecodeError:
        breakdown = {}
    return {
        "season": row["season"],
        "as_of_gw": int(row["as_of_gw"]),
        "n": int(row["n"] or 0),
        "mae": row["mae"],
        "bias": row["bias"],
        "unmatched": row["unmatched"],
        "computed_at": row["computed_at"],
        **breakdown,
    }


CAL_NUDGE_FRAC = 0.5
CAL_NUDGE_CLIP = 0.8
CAL_BIN_MIN_N = 25


def _calibration_for_nudge(
    conn: sqlite3.Connection, season: str, as_of_gw: int
) -> dict[str, Any] | None:
    prev = int(as_of_gw) - 1 if as_of_gw and int(as_of_gw) > 1 else None
    cal = latest_calibration(conn, season, prev) if prev else None
    if not cal:
        cal = latest_calibration(conn, season, None)
    return cal


def _parse_cal_bin_bounds(raw: Any) -> tuple[float | None, float | None]:
    if raw is None:
        return None, None
    text = str(raw)
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return None, None


def _bias_nudge(bias: Any) -> float:
    try:
        v = -CAL_NUDGE_FRAC * float(bias)
    except (TypeError, ValueError):
        return 0.0
    return max(-CAL_NUDGE_CLIP, min(CAL_NUDGE_CLIP, v))


def calibration_nudge(
    cal: dict[str, Any] | None,
    position: str | None,
    xpts_next: float | None = None,
) -> float:
    """Shift this-GW xPts by −½ last bias (position, blended with reliability bin)."""
    if not cal:
        return 0.0
    bias = None
    pos = str(position or "").upper()
    by_pos = cal.get("by_position") or []
    if isinstance(by_pos, dict):
        row = by_pos.get(pos) or by_pos.get(position) or {}
        bias = row.get("bias") if isinstance(row, dict) else None
    elif isinstance(by_pos, list):
        for row in by_pos:
            if not isinstance(row, dict):
                continue
            label = str(row.get("position") or row.get("label") or "").upper()
            if label == pos:
                bias = row.get("bias")
                break
    if bias is None:
        bias = cal.get("bias")
    pos_nudge = _bias_nudge(bias)
    bin_nudge = None
    if xpts_next is not None:
        try:
            x = float(xpts_next)
        except (TypeError, ValueError):
            x = None
        table = cal.get("table") or []
        if x is not None and isinstance(table, list):
            hit = None
            nearest = None
            nearest_d = None
            for row in table:
                if not isinstance(row, dict) or int(row.get("n") or 0) < CAL_BIN_MIN_N:
                    continue
                lo, hi = _parse_cal_bin_bounds(row.get("bin"))
                pm = row.get("pred_mean")
                am = row.get("actual_mean")
                b = row.get("bias")
                if b is None and pm is not None and am is not None:
                    try:
                        b = float(pm) - float(am)
                    except (TypeError, ValueError):
                        b = None
                if b is None:
                    continue
                if lo is not None and hi is not None and lo <= x <= hi:
                    hit = b
                    break
                if pm is not None:
                    d = abs(float(pm) - x)
                    if nearest_d is None or d < nearest_d:
                        nearest_d = d
                        nearest = b
            chosen = hit if hit is not None else nearest
            if chosen is not None:
                bin_nudge = _bias_nudge(chosen)
    if bin_nudge is None:
        return round(pos_nudge, 3)
    return round(0.5 * pos_nudge + 0.5 * bin_nudge, 3)


def _apply_cal_nudge(row: dict[str, Any], cal: dict[str, Any] | None) -> dict[str, Any]:
    nxt = row.get("xpts_next")
    nudge = calibration_nudge(cal, row.get("position"), nxt)
    row["cal_nudge"] = nudge
    if nxt is not None:
        row["xpts_next_raw"] = nxt
        row["xpts_next"] = round(float(nxt) + nudge, 2)
    gws = row.get("xpts_gws")
    if isinstance(gws, list) and gws:
        raw0 = float(gws[0] or 0)
        row["xpts_gws"] = [round(raw0 + nudge, 2)] + [float(x or 0) for x in gws[1:]]
    return row


def last_week_map(
    conn: sqlite3.Connection, season: str, as_of_gw: int, fpl_ids: list[int]
) -> dict[int, dict[str, Any]]:
    pred_gw = int(as_of_gw) - 1
    ids = [int(i) for i in fpl_ids]
    if pred_gw < 1 or not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    actuals: dict[int, int] = {}
    for r in conn.execute(
        f"""
        SELECT fpl_id, COALESCE(SUM(total_points), 0) AS pts
        FROM fact_fpl_player_gw
        WHERE season = ? AND round = ? AND fpl_id IN ({placeholders})
        GROUP BY fpl_id
        """,
        (season, pred_gw, *ids),
    ):
        try:
            actuals[int(r["fpl_id"])] = int(r["pts"] or 0)
        except (TypeError, ValueError):
            continue
    preds: dict[int, float] = {}
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    for table in ("score_player_asof", "score_player_asof_hist"):
        if table not in tables:
            continue
        missing = [i for i in ids if i not in preds]
        if not missing:
            break
        ph = ",".join("?" * len(missing))
        for r in conn.execute(
            f"""
            SELECT fpl_id, json_extract(card_json, '$.xpts_next') AS xpts_next
            FROM {table}
            WHERE season = ? AND as_of_gw = ? AND fpl_id IN ({ph})
            """,
            (season, pred_gw, *missing),
        ):
            if r["xpts_next"] is None:
                continue
            try:
                preds[int(r["fpl_id"])] = round(float(r["xpts_next"]), 2)
            except (TypeError, ValueError):
                continue
    out: dict[int, dict[str, Any]] = {}
    for fid in ids:
        pred = preds.get(fid)
        actual = actuals.get(fid)
        if pred is None and actual is None:
            continue
        out[fid] = {"event": pred_gw, "pred": pred, "actual": actual, "raw": True}
    return out


def last_week_outcome(
    conn: sqlite3.Connection, season: str, as_of_gw: int, fpl_id: int
) -> dict[str, Any] | None:
    return last_week_map(conn, season, as_of_gw, [int(fpl_id)]).get(int(fpl_id))


def template_vs_squad(pool: list[dict[str, Any]], squad_ids: list[int]) -> dict[str, Any]:
    from fpl_model.squad.optimize import SQUAD_NEED

    need = dict(SQUAD_NEED)
    template: list[dict[str, Any]] = []
    for p in sorted(pool, key=lambda x: -(x.get("selected_by_percent") or 0)):
        pos = p.get("position")
        if need.get(pos, 0) <= 0:
            continue
        need[pos] -= 1
        template.append(p)
        if sum(need.values()) == 0:
            break
    t_ids = {int(p["fpl_id"]) for p in template}
    s_ids = {int(i) for i in squad_ids}
    you = [p for p in pool if int(p["fpl_id"]) in s_ids]
    diffs = [p for p in you if int(p["fpl_id"]) not in t_ids]
    own_you = sum(float(p.get("selected_by_percent") or 0) for p in you) / max(1, len(you))
    own_t = sum(float(p.get("selected_by_percent") or 0) for p in template) / max(1, len(template))
    return {
        "overlap": len(t_ids & s_ids),
        "template_n": len(template),
        "template_owned_avg": round(own_t, 1),
        "you_owned_avg": round(own_you, 1),
        "differentials": [
            {
                "fpl_id": p["fpl_id"],
                "web_name": p["web_name"],
                "position": p.get("position"),
                "selected_by_percent": p.get("selected_by_percent"),
            }
            for p in sorted(diffs, key=lambda x: (x.get("selected_by_percent") or 99))[:8]
        ],
        "template": [
            {
                "fpl_id": p["fpl_id"],
                "web_name": p["web_name"],
                "position": p.get("position"),
                "selected_by_percent": p.get("selected_by_percent"),
            }
            for p in template
        ],
    }


def chips_for_squad(
    conn: sqlite3.Connection,
    season: str,
    as_of_gw: int,
    squad_ids: list[int],
    *,
    chips_used: list[str] | None = None,
) -> dict[str, Any]:
    from fpl_model.squad.chips import freehit_indicator, recommend_chips, scan_chip_weeks

    ids = [int(i) for i in squad_ids]
    players = list_players(conn, season, as_of_gw, sort="xpts", ids=ids)
    by_id = {p["fpl_id"]: p for p in players}
    squad = [by_id[i] for i in ids if i in by_id]
    if len(squad) != 15:
        return {"ok": False, "error": "need a full 15", "weeks": [], "chips": []}
    scan = scan_chip_weeks(squad, as_of_gw=as_of_gw, chips_used=chips_used)
    fh_pool = list_players(conn, season, as_of_gw, sort="xpts_next", limit=500)
    scan["freehit"] = freehit_indicator(squad, fh_pool, as_of_gw=as_of_gw)
    return {
        "ok": True,
        "season": season,
        "as_of_gw": as_of_gw,
        "chips": recommend_chips(squad, chips_used=chips_used, as_of_gw=as_of_gw),
        **scan,
    }


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
        "xpts_next": _round(r["xpts_next"], 2) if "xpts_next" in r.keys() else None,
        "xpts_gws": _parse_xpts_gws(r["xpts_gws"]) if "xpts_gws" in r.keys() else None,
        "flags": _parse_flags(r["flags"]) if "flags" in r.keys() else None,
        "minutes_risk": _parse_flags(r["minutes_risk"]) if "minutes_risk" in r.keys() else None,
        "availability": _round(r["availability"], 3) if "availability" in r.keys() else None,
        "news": r["news"] if "news" in r.keys() else None,
    }


def _parse_xpts_gws(raw: Any) -> list[float] | None:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return [float(x or 0) for x in raw]
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, list):
            return [float(x or 0) for x in parsed]
    return None


def _parse_flags(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None
