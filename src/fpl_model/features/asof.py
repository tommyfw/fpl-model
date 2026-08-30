from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd

from fpl_model.config import Settings, load_settings
from fpl_model.db import connect
from fpl_model.features.stats import per90, shrink_to_mean

XP_FORBIDDEN = "xP"


def load_gw(conn: sqlite3.Connection, season: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM fact_fpl_player_gw WHERE season = ?",
        conn,
        params=(season,),
    )


def load_fixtures(conn: sqlite3.Connection, season: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM fact_fixture WHERE season = ?",
        conn,
        params=(season,),
    )


def load_players(conn: sqlite3.Connection, season: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM dim_player WHERE season = ?",
        conn,
        params=(season,),
    )


def load_teams(conn: sqlite3.Connection, season: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM dim_team WHERE season = ?",
        conn,
        params=(season,),
    )


def _asof_cutoff(fx: pd.DataFrame, as_of_gw: int) -> pd.Timestamp | None:
    kicks = pd.to_datetime(fx.loc[fx["event"] == as_of_gw, "kickoff_time"], utc=True, errors="coerce")
    if kicks.notna().any():
        return kicks.min()
    return None


def _apply_understat(
    conn: sqlite3.Connection,
    season: str,
    as_of_gw: int,
    fx: pd.DataFrame,
    out: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    cutoff = _asof_cutoff(fx, as_of_gw)
    if cutoff is None:
        return out
    cut_date = cutoff.tz_convert("UTC").strftime("%Y-%m-%d") if getattr(cutoff, "tzinfo", None) else cutoff.strftime("%Y-%m-%d")

    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_understat_player_match'"
    ).fetchone()
    if not exists:
        return out

    us = pd.read_sql_query(
        """
        SELECT m.fpl_id, u.*
        FROM fact_understat_player_match u
        JOIN map_player_ids m
          ON m.season = u.season AND m.understat_id = u.understat_id
        WHERE u.season = ? AND m.fpl_id IS NOT NULL AND u.date < ?
        """,
        conn,
        params=(season, cut_date),
    )
    if not us.empty:
        g = us.groupby("fpl_id", as_index=True).agg(
            us_minutes=("minutes", "sum"),
            us_npxg=("npxg", "sum"),
            us_npxa=("xa", "sum"),
            us_shots=("shots", "sum"),
            us_sib=("shots_in_box", "sum"),
            us_kp=("key_passes", "sum"),
            us_xgchain=("xgchain", "sum"),
            us_xgbuildup=("xgbuildup", "sum"),
        )
        pos_mode = us.groupby("fpl_id")["position"].agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else None)
        g["us_position"] = pos_mode
        prior = settings.shrink_prior_minutes
        idx = g.index.intersection(out.index)
        if len(idx):
            minutes = g.loc[idx, "us_minutes"]
            for dest, src in [
                ("npxg90", "us_npxg"),
                ("npxa90", "us_npxa"),
                ("shots90", "us_shots"),
                ("shots_in_box90", "us_sib"),
                ("key_passes90", "us_kp"),
                ("xgchain90", "us_xgchain"),
            ]:
                raw = pd.Series(per90(g.loc[idx, src], minutes), index=idx)
                pos_mean = float(raw.mean()) if raw.notna().any() else 0.0
                out.loc[idx, dest] = shrink_to_mean(raw, minutes, pos_mean, prior)
            out.loc[idx, "npxgi90"] = out.loc[idx, "npxg90"].fillna(0) + out.loc[idx, "npxa90"].fillna(0)
            shots = g.loc[idx, "us_shots"].clip(lower=1)
            out.loc[idx, "chance_quality90"] = g.loc[idx, "us_npxg"] / shots
            out.loc[idx, "us_position"] = g.loc[idx, "us_position"]
            out.loc[idx, "understat_matched"] = 1

    teams = load_teams(conn, season)
    if not teams.empty:
        from fpl_model.ingest.id_map import team_key

        teams = teams.copy()
        teams["tkey"] = teams["name"].map(team_key)
        ut = pd.read_sql_query(
            "SELECT * FROM fact_understat_team_match WHERE season = ? AND finished = 1 AND date < ?",
            conn,
            params=(season, cut_date),
        )
        if not ut.empty:
            ut = ut.merge(teams[["tkey", "team_id"]], left_on="team_key", right_on="tkey", how="left")
            tg = ut.dropna(subset=["team_id"]).groupby("team_id", as_index=True).agg(
                n=("match_id", "nunique"),
                xgf=("xg_for", "sum"),
                xga=("xg_against", "sum"),
            )
            tg["n"] = tg["n"].clip(lower=1)
            atk = tg["xgf"] / tg["n"]
            de = -(tg["xga"] / tg["n"])
            out["team_attack"] = out["team_id"].map(atk).fillna(out.get("team_attack"))
            out["team_defence"] = out["team_id"].map(de).fillna(out.get("team_defence"))
            out["p_clean_sheet"] = _cs_rate((-out["team_defence"]).clip(lower=0.05))
    return out


def asof_gw_frame(gw: pd.DataFrame, as_of_gw: int) -> pd.DataFrame:
    """Rows strictly before gameweek T. Never includes round >= T."""
    if XP_FORBIDDEN in gw.columns:
        gw = gw.drop(columns=[XP_FORBIDDEN])
    return gw.loc[gw["round"] < as_of_gw].copy()


def upcoming_fixtures(fx: pd.DataFrame, as_of_gw: int, horizon: int) -> pd.DataFrame:
    """Fixtures in T .. T+horizon-1. Scores ignored for scoring (may be present in archive)."""
    mask = fx["event"].between(as_of_gw, as_of_gw + horizon - 1)
    out = fx.loc[mask].copy()
    return out


def _team_matches(fx: pd.DataFrame, as_of_gw: int) -> pd.DataFrame:
    played = fx.loc[fx["event"] < as_of_gw].copy()
    rows = []
    for _, r in played.iterrows():
        if pd.isna(r["event"]):
            continue
        rows.append({"team_id": r["team_h"], "event": r["event"], "gf": r["team_h_score"], "ga": r["team_a_score"], "xg_for": np.nan, "home": 1})
        rows.append({"team_id": r["team_a"], "event": r["event"], "gf": r["team_a_score"], "ga": r["team_h_score"], "xg_for": np.nan, "home": 0})
    if not rows:
        return pd.DataFrame(columns=["team_id", "event", "gf", "ga", "matches"])
    tm = pd.DataFrame(rows)
    g = tm.groupby("team_id", as_index=False).agg(
        matches=("event", "nunique"),
        gf=("gf", "sum"),
        ga=("ga", "sum"),
    )
    g["gf90"] = g["gf"] / g["matches"].clip(lower=1)
    g["ga90"] = g["ga"] / g["matches"].clip(lower=1)
    return g


def _cs_rate(ga90: pd.Series) -> pd.Series:
    # Poisson P(0 | lambda=ga90)
    lam = ga90.clip(lower=0.05)
    return np.exp(-lam)


def _defcon_hits(past: pd.DataFrame, thresholds: dict) -> pd.Series:
    pos = past["position"]
    cbit = past["clearances_blocks_interceptions"].fillna(0) + past["tackles"].fillna(0)
    rec = past["recoveries"].fillna(0)
    dc = past["defensive_contribution"]
    hits = []
    for i, row in past.iterrows():
        p = row.get("position")
        thr = thresholds.get(p)
        if not thr:
            hits.append(0)
            continue
        val = dc.loc[i] if pd.notna(dc.loc[i]) else (cbit.loc[i] + (rec.loc[i] if p in {"MID", "FWD"} else 0))
        hits.append(int(val >= thr))
    return pd.Series(hits, index=past.index)


def build_features_for_asof(
    conn: sqlite3.Connection,
    season: str,
    as_of_gw: int,
    settings: Settings | None = None,
) -> pd.DataFrame:
    settings = settings or load_settings()
    gw = load_gw(conn, season)
    fx = load_fixtures(conn, season)
    players = load_players(conn, season)
    past = asof_gw_frame(gw, as_of_gw)
    if past.empty:
        return pd.DataFrame()

    past = past.merge(players[["fpl_id", "position", "team_id", "web_name", "status", "chance_of_playing_next", "penalties_order", "corners_order", "direct_fk_order"]], on="fpl_id", how="left")

    team_stats = _team_matches(fx, as_of_gw)
    past_n = past.groupby("fpl_id", as_index=False).agg(
        minutes=("minutes", "sum"),
        starts=("starts", "sum"),
        points=("total_points", "sum"),
        goals=("goals_scored", "sum"),
        assists=("assists", "sum"),
        bonus=("bonus", "sum"),
        bps=("bps", "sum"),
        cs=("clean_sheets", "sum"),
        gc=("goals_conceded", "sum"),
        saves=("saves", "sum"),
        xg=("expected_goals", "sum"),
        xa=("expected_assists", "sum"),
        xgi=("expected_goal_involvements", "sum"),
        xgc=("expected_goals_conceded", "sum"),
        threat=("threat", "sum"),
        creativity=("creativity", "sum"),
        appearances=("round", "nunique"),
        last_value=("value", "last"),
        last_selected=("selected", "last"),
        last_round=("round", "max"),
    )
    # last() is order-dependent; sort first
    past_sorted = past.sort_values(["fpl_id", "round"])
    last_vals = past_sorted.groupby("fpl_id").agg(last_value=("value", "last"), last_selected=("selected", "last"), last_round=("round", "max"))
    past_n = past_n.drop(columns=["last_value", "last_selected", "last_round"]).merge(last_vals, on="fpl_id")

    def _window_sum(df: pd.DataFrame, n: int, col: str) -> pd.Series:
        recent_rounds = sorted(df["round"].dropna().unique())
        keep = set(recent_rounds[-n:]) if recent_rounds else set()
        sub = df.loc[df["round"].isin(keep)]
        return sub.groupby("fpl_id")[col].sum()

    last5_pts = _window_sum(past, 5, "total_points")
    last10_pts = _window_sum(past, 10, "total_points")
    last5_min = _window_sum(past, 5, "minutes")
    last10_min = _window_sum(past, 10, "minutes")
    last5_starts = _window_sum(past, 5, "starts")
    last10_starts = _window_sum(past, 10, "starts")

    meta = players.set_index("fpl_id")
    out = past_n.set_index("fpl_id")
    out["position"] = meta["position"]
    out["team_id"] = meta["team_id"]
    out["web_name"] = meta["web_name"]
    out["status"] = meta["status"]
    out["chance_of_playing_next"] = meta["chance_of_playing_next"]
    out["penalties_order"] = meta["penalties_order"]
    out["corners_order"] = meta["corners_order"]
    out["direct_fk_order"] = meta["direct_fk_order"]

    out = out.join(team_stats.set_index("team_id"), on="team_id")
    out["matches_available"] = out["matches"].fillna(0).clip(lower=1)
    out["start_pct"] = out["starts"] / out["matches_available"]
    out["minutes_pct"] = out["minutes"] / (90.0 * out["matches_available"])
    out["start_pct_l5"] = last5_starts.reindex(out.index).fillna(0) / 5.0
    out["start_pct_l10"] = last10_starts.reindex(out.index).fillna(0) / 10.0
    out["minutes_pct_l5"] = last5_min.reindex(out.index).fillna(0) / (90.0 * 5)
    out["minutes_pct_l10"] = last10_min.reindex(out.index).fillna(0) / (90.0 * 10)

    prior = settings.shrink_prior_minutes
    for col, src in [
        ("points90", "points"),
        ("goals90", "goals"),
        ("assists90", "assists"),
        ("bonus90", "bonus"),
        ("bps90", "bps"),
        ("npxg90", "xg"),
        ("npxa90", "xa"),
        ("npxgi90", "xgi"),
        ("xgc90", "xgc"),
        ("saves90", "saves"),
        ("threat90", "threat"),
        ("creativity90", "creativity"),
    ]:
        raw = pd.Series(per90(out[src], out["minutes"]), index=out.index)
        pos_means = raw.groupby(out["position"]).transform("mean").fillna(raw.mean())
        out[col] = shrink_to_mean(raw, out["minutes"], float(pos_means.mean()) if pos_means.empty else 0, prior)
        # shrink per row toward its position mean
        out[col] = shrink_to_mean(raw, out["minutes"], 0.0, prior)
        for pos, mean in raw.groupby(out["position"]).mean().items():
            idx = out["position"] == pos
            out.loc[idx, col] = shrink_to_mean(raw[idx], out.loc[idx, "minutes"], float(mean) if pd.notna(mean) else 0.0, prior)

    form_w = (0.6, 0.3, 0.1)
    pts_l5 = last5_pts.reindex(out.index).fillna(0)
    last10_pts = last10_pts.reindex(out.index).fillna(0)
    out["points_form"] = form_w[0] * pts_l5 + form_w[1] * last10_pts + form_w[1] * 0 + form_w[2] * out["points"]
    # rescale form to per-gw-ish
    out["points_form"] = (
        form_w[0] * (pts_l5 / 5.0)
        + form_w[1] * (last10_pts.reindex(out.index).fillna(0) / 10.0)
        + form_w[2] * (out["points"] / out["matches_available"])
    )

    past = past.copy()
    past["defcon_hit"] = _defcon_hits(past, settings.defcon_threshold)
    started = past.loc[past["starts"] >= 1]
    hit_rate = started.groupby("fpl_id")["defcon_hit"].mean()
    out["defcon_hit_rate"] = hit_rate.reindex(out.index).fillna(0.0)

    # proxies until Understat overlays
    out["shots_in_box90"] = out["threat90"] / 10.0
    out["key_passes90"] = out["creativity90"] / 10.0
    out["chance_quality90"] = out["npxg90"]
    out["xgchain90"] = out["npxgi90"]
    out["understat_matched"] = 0
    out["p_clean_sheet"] = _cs_rate(out["ga90"].fillna(out["ga90"].median() if out["ga90"].notna().any() else 1.2))
    out["team_attack"] = out["gf90"]
    out["team_defence"] = -out["ga90"]
    out = _apply_understat(conn, season, as_of_gw, fx, out, settings)
    out["price"] = out["last_value"] / 10.0
    out["availability"] = np.where(
        out["status"].isin(["i", "s", "u", "n"]),
        0.1,
        np.where(out["status"] == "d", out["chance_of_playing_next"].fillna(50) / 100.0, 1.0),
    )
    out["pen_taker"] = (out["penalties_order"] == 1).astype(float)
    out["set_piece_taker"] = (
        (out["corners_order"].fillna(99) <= 2) | (out["direct_fk_order"].fillna(99) <= 2)
    ).astype(float)

    # upcoming fixtures per team
    upc = upcoming_fixtures(fx, as_of_gw, settings.horizon_gws)
    team_lookup = out.groupby("team_id")[["team_attack", "team_defence"]].first()
    fixture_rows = []
    for fpl_id, row in out.iterrows():
        team = row["team_id"]
        fxs = upc.loc[(upc["team_h"] == team) | (upc["team_a"] == team)]
        by_event: dict[int, list] = {}
        for _, fr in fxs.iterrows():
            home = fr["team_h"] == team
            opp = fr["team_a"] if home else fr["team_h"]
            opp_att = team_lookup["team_attack"].get(opp, team_lookup["team_attack"].mean() if len(team_lookup) else 1.2)
            opp_def_score = team_lookup["team_defence"].get(opp, team_lookup["team_defence"].mean() if len(team_lookup) else -1.2)
            leak = float(-opp_def_score) if pd.notna(opp_def_score) else 1.2
            att = float(opp_att) if pd.notna(opp_att) else 1.2
            ev = int(fr["event"]) if pd.notna(fr["event"]) else as_of_gw
            by_event.setdefault(ev, []).append(
                {
                    "home": 1.0 if home else 0.0,
                    "opp_attack": att,
                    "opp_defence_leak": leak,
                    "p_cs": float(np.exp(-att)),
                }
            )
        fixture_rows.append({"fpl_id": fpl_id, "fixtures_json": json.dumps({str(k): v for k, v in by_event.items()})})
    fxj = pd.DataFrame(fixture_rows).set_index("fpl_id")
    out["fixtures_json"] = fxj["fixtures_json"]

    out = out.reset_index()
    return out


def persist_features(conn: sqlite3.Connection, season: str, as_of_gw: int, feats: pd.DataFrame) -> None:
    conn.execute("DELETE FROM feat_player_asof WHERE season = ? AND as_of_gw = ?", (season, as_of_gw))
    recs = []
    feats = feats.copy()
    if "fpl_id" not in feats.columns:
        feats = feats.reset_index()
        if "index" in feats.columns and "fpl_id" not in feats.columns:
            feats = feats.rename(columns={"index": "fpl_id"})
    for _, r in feats.iterrows():
        payload = r.drop(labels=["fpl_id", "position", "team_id", "price"], errors="ignore").to_dict()
        recs.append(
            (
                season,
                int(as_of_gw),
                int(r["fpl_id"]),
                r.get("position"),
                int(r["team_id"]) if pd.notna(r.get("team_id")) else None,
                float(r["price"]) if pd.notna(r.get("price")) else None,
                json.dumps(payload, default=str),
            )
        )
    conn.executemany(
        """INSERT INTO feat_player_asof(season, as_of_gw, fpl_id, position, team_id, price, features_json)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        recs,
    )
    conn.commit()


def build_and_store(season: str, as_of_gw: int, settings: Settings | None = None) -> int:
    settings = settings or load_settings()
    conn = connect(settings)
    feats = build_features_for_asof(conn, season, as_of_gw, settings)
    persist_features(conn, season, as_of_gw, feats)
    n = len(feats)
    conn.close()
    return n
