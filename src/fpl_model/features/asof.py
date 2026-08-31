from __future__ import annotations

import json
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from fpl_model.config import Settings, load_settings
from fpl_model.db import connect
from fpl_model.features.rotation import competing_start_rates
from fpl_model.features.stats import blend_playing_time, per90, shrink_to_mean
from fpl_model.ingest.priors import load_player_priors, load_team_priors
from fpl_model.scoring.clean_sheet import horizon_p_cs, pack_fixture
from fpl_model.scoring.role import defcon_count, penalty_score, pitch_attack_score, pitch_central_score, set_piece_score

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
        pos_played = us.loc[us["position"].notna() & (us["position"].astype(str) != "Sub")]
        pos_mode = pos_played.groupby("fpl_id")["position"].agg(
            lambda s: s.mode().iloc[0] if not s.empty else None
        )
        g["us_position"] = pos_mode
        prior = settings.shrink_prior_minutes
        idx = g.index.intersection(out.index)
        if len(idx):
            minutes = g.loc[idx, "us_minutes"]
            # Season totals from Understat (same as-of window as the rates).
            out.loc[idx, "npxg"] = g.loc[idx, "us_npxg"].astype(float)
            out.loc[idx, "npxa"] = g.loc[idx, "us_npxa"].astype(float)
            out.loc[idx, "us_minutes"] = minutes.astype(float)
            for dest, src, obs_dest in [
                ("npxg90", "us_npxg", "npxg90_obs"),
                ("npxa90", "us_npxa", "npxa90_obs"),
                ("shots90", "us_shots", None),
                ("shots_in_box90", "us_sib", None),
                ("key_passes90", "us_kp", None),
                ("xgchain90", "us_xgchain", None),
            ]:
                raw = pd.Series(per90(g.loc[idx, src], minutes), index=idx).astype(float)
                if obs_dest:
                    out.loc[idx, obs_dest] = raw
                pos_mean = float(raw.mean()) if raw.notna().any() else 0.0
                out.loc[idx, dest] = shrink_to_mean(
                    raw, minutes, pos_mean, prior, max_prior_weight=settings.max_prior_weight
                ).astype(float)
            out.loc[idx, "npxgi90"] = out.loc[idx, "npxg90"].fillna(0) + out.loc[idx, "npxa90"].fillna(0)
            shots = g.loc[idx, "us_shots"].clip(lower=1)
            out.loc[idx, "chance_quality90"] = (g.loc[idx, "us_npxg"] / shots).astype(float)
            out.loc[idx, "us_position"] = g.loc[idx, "us_position"]
            out.loc[idx, "understat_matched"] = 1
            out.loc[idx, "xg_source"] = "understat"
            last5 = us.sort_values(["fpl_id", "date"]).groupby("fpl_id", as_index=False).tail(5)
            g5 = last5.groupby("fpl_id").agg(
                m5=("minutes", "sum"),
                npxg5=("npxg", "sum"),
                npa5=("xa", "sum"),
            )
            raw_l5 = pd.Series(per90(g5["npxg5"] + g5["npa5"], g5["m5"]), index=g5.index)
            raw_l5 = raw_l5.reindex(idx)
            m5 = g5["m5"].reindex(idx).fillna(0)
            season_gi = out.loc[idx, "npxgi90"].astype(float)
            out.loc[idx, "npxgi90_l5"] = shrink_to_mean(
                raw_l5.fillna(season_gi),
                m5,
                season_gi.fillna(0.0),
                180,
                max_prior_weight=settings.max_prior_weight,
            ).astype(float)

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
        atk = pd.Series(dtype=float)
        de = pd.Series(dtype=float)
        n_obs = pd.Series(dtype=float)
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
            n_obs = tg["n"]
            trends = _xg_trend_by_team(ut)
            if not trends.empty:
                tmap_a = trends.set_index("team_id")["team_attack_trend"]
                tmap_d = trends.set_index("team_id")["team_defence_trend"]
                out["team_attack_trend"] = out["team_id"].map(tmap_a).fillna(0.0)
                out["team_defence_trend"] = out["team_id"].map(tmap_d).fillna(0.0)
        # Always blend with prior-season xG (promoted sides → previous PL average).
        atk, de = _blend_team_strength(conn, season, atk, de, n_obs, settings)
        out["team_attack"] = out["team_id"].map(atk).fillna(out.get("team_attack"))
        out["team_defence"] = out["team_id"].map(de).fillna(out.get("team_defence"))
        out["p_clean_sheet"] = _cs_rate((-out["team_defence"]).clip(lower=0.05))
    if "team_attack_trend" not in out.columns:
        out["team_attack_trend"] = 0.0
        out["team_defence_trend"] = 0.0

    if not us.empty:
        played = us.loc[us["position"].notna() & (us["position"].astype(str) != "Sub")].copy()
        if not played.empty:
            played["pitch_atk"] = played["position"].map(pitch_attack_score)
            played = played.sort_values(["fpl_id", "date"])
            season_atk = played.groupby("fpl_id")["pitch_atk"].mean()
            l5_atk = played.groupby("fpl_id").tail(5).groupby("fpl_id")["pitch_atk"].mean()
            idx = season_atk.index.intersection(out.index)
            out.loc[idx, "pitch_attack_l5"] = l5_atk.reindex(idx)
            out.loc[idx, "role_change_raw"] = (l5_atk - season_atk).reindex(idx).fillna(0.0)
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


def _blend_team_strength(
    conn: sqlite3.Connection,
    season: str,
    atk: pd.Series,
    de: pd.Series,
    n: pd.Series,
    settings: Settings,
) -> tuple[pd.Series, pd.Series]:
    """Shrink observed team xG rates toward prior-season (promoted → PL average)."""
    priors = load_team_priors(conn, season)
    if priors.empty:
        return atk, de
    prior_n = max(int(getattr(settings, "team_prior_matches", 8)), 1)
    ids = priors.index.union(atk.index).union(de.index)
    obs_n = n.reindex(ids).fillna(0.0).astype(float)
    w = obs_n / (obs_n + prior_n)
    pa = priors["prior_attack"].reindex(ids).fillna(float(priors["prior_attack"].mean()))
    pde = priors["prior_defence"].reindex(ids).fillna(float(priors["prior_defence"].mean()))
    oa = atk.reindex(ids)
    od = de.reindex(ids)
    blended_atk = pd.Series(np.where(oa.notna(), w * oa.to_numpy() + (1 - w) * pa.to_numpy(), pa.to_numpy()), index=ids)
    blended_de = pd.Series(np.where(od.notna(), w * od.to_numpy() + (1 - w) * pde.to_numpy(), pde.to_numpy()), index=ids)
    return blended_atk, blended_de


def _xg_trend_by_team(ut: pd.DataFrame, last_n: int = 8) -> pd.DataFrame:
    """Recent vs earlier xG/match. Defence trend is a drop in xGA (positive = improving)."""
    if ut.empty or "team_id" not in ut.columns:
        return pd.DataFrame(columns=["team_id", "team_attack_trend", "team_defence_trend"])
    ut = ut.dropna(subset=["team_id"]).copy()
    ut["date"] = ut["date"].astype(str)
    ut = ut.sort_values(["team_id", "date"])
    rows = []
    for tid, g in ut.groupby("team_id"):
        n = len(g)
        k = min(last_n, max(3, n // 2))
        if n < 6 or k < 3:
            rows.append({"team_id": tid, "team_attack_trend": 0.0, "team_defence_trend": 0.0})
            continue
        recent = g.tail(k)
        earlier = g.iloc[: n - k]
        if earlier.empty:
            rows.append({"team_id": tid, "team_attack_trend": 0.0, "team_defence_trend": 0.0})
            continue
        rows.append(
            {
                "team_id": tid,
                "team_attack_trend": float(recent["xg_for"].mean() - earlier["xg_for"].mean()),
                "team_defence_trend": float(earlier["xg_against"].mean() - recent["xg_against"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _price_change(past_sorted: pd.DataFrame, lag_gws: int = 5) -> pd.Series:
    """Current value minus value ~lag_gws ago, in £m."""
    rows: dict[int, float] = {}
    for fid, g in past_sorted.groupby("fpl_id"):
        last = float(g["value"].iloc[-1])
        cutoff = int(g["round"].iloc[-1]) - lag_gws
        older = g.loc[g["round"] <= cutoff]
        prev = float(older["value"].iloc[-1]) if not older.empty else float(g["value"].iloc[0])
        rows[int(fid)] = (last - prev) / 10.0
    return pd.Series(rows)


def _defcon_hits(past: pd.DataFrame, thresholds: dict) -> pd.Series:
    counts = past.apply(defcon_count, axis=1)
    thr = past["position"].map(lambda p: thresholds.get(p) or 10**9)
    started = past["starts"].fillna(0) >= 1
    return ((counts >= thr) & started).astype(int)


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
    if players.empty:
        return pd.DataFrame()

    pcols = [
        "fpl_id",
        "position",
        "team_id",
        "web_name",
        "status",
        "chance_of_playing_next",
        "penalties_order",
        "corners_order",
        "direct_fk_order",
    ]
    for extra in ("code", "now_cost", "selected_by_percent"):
        if extra in players.columns:
            pcols.append(extra)

    def _window_sum(df: pd.DataFrame, n: int, col: str) -> pd.Series:
        recent_rounds = sorted(df["round"].dropna().unique())
        keep = set(recent_rounds[-n:]) if recent_rounds else set()
        sub = df.loc[df["round"].isin(keep)]
        return sub.groupby("fpl_id")[col].sum()

    if not past.empty:
        if "position" in past.columns:
            past = past.drop(columns=["position"])
        past = past.merge(players[pcols], on="fpl_id", how="left")
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
        )
        past_sorted = past.sort_values(["fpl_id", "round"])
        last_vals = past_sorted.groupby("fpl_id").agg(
            last_value=("value", "last"), last_selected=("selected", "last"), last_round=("round", "max")
        )
        past_n = past_n.merge(last_vals, on="fpl_id")
        last5_pts = _window_sum(past, 5, "total_points")
        last10_pts = _window_sum(past, 10, "total_points")
        last5_min = _window_sum(past, 5, "minutes")
        last10_min = _window_sum(past, 10, "minutes")
        last5_starts = _window_sum(past, 5, "starts")
        last10_starts = _window_sum(past, 10, "starts")
        meta = players.set_index("fpl_id")
        out = past_n.set_index("fpl_id")
        for c in ["position", "team_id", "web_name", "status", "chance_of_playing_next", "penalties_order", "corners_order", "direct_fk_order"]:
            out[c] = meta[c]
    else:
        out = players.set_index("fpl_id").copy()
        for c in [
            "minutes",
            "starts",
            "points",
            "goals",
            "assists",
            "bonus",
            "bps",
            "cs",
            "gc",
            "saves",
            "xg",
            "xa",
            "xgi",
            "xgc",
            "threat",
            "creativity",
            "appearances",
        ]:
            out[c] = 0
        out["last_value"] = pd.to_numeric(out.get("now_cost"), errors="coerce").fillna(0)
        out["last_selected"] = pd.to_numeric(out.get("selected_by_percent"), errors="coerce").fillna(0)
        last5_pts = last10_pts = last5_min = last10_min = last5_starts = last10_starts = pd.Series(dtype=float)
        past_sorted = pd.DataFrame(columns=["fpl_id", "round", "value"])

    missing = players.set_index("fpl_id").index.difference(out.index)
    if len(missing):
        extra = players.set_index("fpl_id").loc[missing].copy()
        for c in [
            "minutes",
            "starts",
            "points",
            "goals",
            "assists",
            "bonus",
            "bps",
            "cs",
            "gc",
            "saves",
            "xg",
            "xa",
            "xgi",
            "xgc",
            "threat",
            "creativity",
            "appearances",
        ]:
            extra[c] = 0
        extra["last_value"] = pd.to_numeric(extra.get("now_cost"), errors="coerce").fillna(0)
        extra["last_selected"] = pd.to_numeric(extra.get("selected_by_percent"), errors="coerce").fillna(0)
        out = pd.concat([out, extra])

    team_stats = _team_matches(fx, as_of_gw)
    out = out.join(team_stats.set_index("team_id"), on="team_id")
    out["matches_available"] = out["matches"].fillna(0).clip(lower=1)
    out["start_pct"] = out["starts"] / out["matches_available"]
    out["minutes_pct"] = out["minutes"] / (90.0 * out["matches_available"])
    n_rounds = int(past["round"].nunique()) if not past.empty else 0
    n5 = max(1, min(5, n_rounds))
    n10 = max(1, min(10, n_rounds))
    out["start_pct_l5"] = last5_starts.reindex(out.index).fillna(0) / n5
    out["start_pct_l10"] = last10_starts.reindex(out.index).fillna(0) / n10
    out["minutes_pct_l5"] = last5_min.reindex(out.index).fillna(0) / (90.0 * n5)
    out["minutes_pct_l10"] = last10_min.reindex(out.index).fillna(0) / (90.0 * n10)

    if not past.empty:
        rot = competing_start_rates(past)
        for c in rot.columns:
            out[c] = rot[c].reindex(out.index)
    if "rotation_start_pct" not in out.columns:
        out["rotation_start_pct"] = np.nan
        out["rival_games"] = 0
        out["pos_slot_rank"] = 1.0
        out["slot_security"] = 1.0
    out["rotation_start_pct"] = pd.to_numeric(out["rotation_start_pct"], errors="coerce")
    low_sample = out.get("rival_games", 0)
    low_sample = pd.to_numeric(low_sample, errors="coerce").fillna(0) < 3
    out.loc[low_sample, "rotation_start_pct"] = out.loc[low_sample, "start_pct_l5"]
    out["rotation_start_pct"] = out["rotation_start_pct"].fillna(out["start_pct_l5"]).clip(0.0, 1.0)
    out["slot_security"] = pd.to_numeric(out.get("slot_security"), errors="coerce").fillna(1.0).clip(0.4, 1.0)
    out["pos_slot_rank"] = pd.to_numeric(out.get("pos_slot_rank"), errors="coerce").fillna(1.0)
    out["rival_games"] = pd.to_numeric(out.get("rival_games"), errors="coerce").fillna(0)

    player_priors = load_player_priors(conn, season)
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
        pos_mean = raw.groupby(out["position"]).transform("mean")
        if pos_mean.isna().all():
            pos_mean = pd.Series(0.0, index=out.index)
        else:
            pos_mean = pos_mean.fillna(float(raw.mean()) if raw.notna().any() else 0.0)
        if not player_priors.empty and col in player_priors.columns:
            blend = player_priors[col].reindex(out.index).fillna(pos_mean)
        else:
            blend = pos_mean
        out[col] = shrink_to_mean(
            raw, out["minutes"], blend, prior, max_prior_weight=settings.max_prior_weight
        )

    if not player_priors.empty:
        matches = out["matches"].fillna(0) if "matches" in out.columns else pd.Series(0.0, index=out.index)
        for c in ("start_pct", "minutes_pct"):
            if c in player_priors.columns:
                out[c] = blend_playing_time(
                    out[c],
                    out["minutes"],
                    player_priors[c],
                    prior,
                    settings.max_playing_prior_weight,
                    out["starts"],
                    matches,
                    settings.prior_playing_grace_matches,
                )

    form_w = (0.6, 0.3, 0.1)
    pts_l5 = last5_pts.reindex(out.index).fillna(0)
    last10_pts = last10_pts.reindex(out.index).fillna(0)
    out["points_form"] = (
        form_w[0] * (pts_l5 / 5.0)
        + form_w[1] * (last10_pts.reindex(out.index).fillna(0) / 10.0)
        + form_w[2] * (out["points"] / out["matches_available"])
    )

    if not past.empty:
        past = past.copy()
        past["defcon_hit"] = _defcon_hits(past, settings.defcon_threshold)
        started = past.loc[past["starts"] >= 1]
        hit_rate = started.groupby("fpl_id")["defcon_hit"].mean()
        out["defcon_hit_rate"] = hit_rate.reindex(out.index).fillna(0.0)
    else:
        out["defcon_hit_rate"] = 0.0

    # proxies until Understat overlays
    out["shots_in_box90"] = out["threat90"] / 10.0
    out["key_passes90"] = out["creativity90"] / 10.0
    out["chance_quality90"] = out["npxg90"]
    out["xgchain90"] = out["npxgi90"]
    out["understat_matched"] = 0
    if "ga90" not in out.columns:
        out["ga90"] = np.nan
        out["gf90"] = np.nan
    out["p_clean_sheet"] = _cs_rate(out["ga90"].fillna(out["ga90"].median() if out["ga90"].notna().any() else 1.2))
    out["team_attack"] = out["gf90"]
    out["team_defence"] = -out["ga90"]
    # FPL season xG/xA as fallback totals until Understat overlays.
    out["npxg"] = pd.to_numeric(out.get("xg"), errors="coerce").astype(float)
    out["npxa"] = pd.to_numeric(out.get("xa"), errors="coerce").astype(float)
    out["xg_source"] = "fpl"
    out["npxg90_obs"] = pd.to_numeric(out["npxg90"], errors="coerce").astype(float)
    out["npxa90_obs"] = pd.to_numeric(out["npxa90"], errors="coerce").astype(float)
    out["us_minutes"] = np.nan
    out = _apply_understat(conn, season, as_of_gw, fx, out, settings)
    from fpl_model.ingest.fbref import attach_fbref

    out = attach_fbref(conn, season, out)
    if "team_attack_trend" not in out.columns:
        out["team_attack_trend"] = 0.0
        out["team_defence_trend"] = 0.0
    if "role_change_raw" not in out.columns:
        out["role_change_raw"] = 0.0
    else:
        out["role_change_raw"] = pd.to_numeric(out["role_change_raw"], errors="coerce").fillna(0.0)
    out["price"] = out["last_value"] / 10.0
    price_delta = _price_change(past_sorted)
    out["price_change"] = price_delta.reindex(out.index).fillna(0.0)
    out["availability"] = np.where(
        out["status"].isin(["i", "s", "u", "n"]),
        0.1,
        np.where(out["status"] == "d", out["chance_of_playing_next"].fillna(50) / 100.0, 1.0),
    )
    out["pen_taker"] = (out["penalties_order"] == 1).astype(float)
    out["set_piece_taker"] = (
        (out["corners_order"].fillna(99) <= 2) | (out["direct_fk_order"].fillna(99) <= 2)
    ).astype(float)
    if "us_position" not in out.columns:
        out["us_position"] = None
    out["pitch_attack"] = out["us_position"].map(lambda p: pitch_attack_score(p))
    out["pitch_central"] = out["us_position"].map(lambda p: pitch_central_score(p))
    out["pen_score"] = out["penalties_order"].map(penalty_score)
    out["sp_score"] = [
        set_piece_score(c, d) for c, d in zip(out["corners_order"], out["direct_fk_order"])
    ]
    p90_l5 = pd.Series(
        per90(last5_pts.reindex(out.index).fillna(0), last5_min.reindex(out.index).fillna(0)),
        index=out.index,
    )
    p90_l10 = pd.Series(
        per90(last10_pts.reindex(out.index).fillna(0), last10_min.reindex(out.index).fillna(0)),
        index=out.index,
    )
    out["points90_blend"] = 0.6 * p90_l5 + 0.3 * p90_l10 + 0.1 * out["points90"]

    # upcoming fixtures per team — P(CS) uses own xGA × opponent xG × H/A, not season GA alone
    upc = upcoming_fixtures(fx, as_of_gw, settings.horizon_gws)
    team_lookup = out.groupby("team_id")[["team_attack", "team_defence"]].first()
    atk_mean = team_lookup["team_attack"].dropna()
    league_xg = float(atk_mean.mean()) if len(atk_mean) else 1.35
    if not np.isfinite(league_xg) or league_xg <= 0.4:
        league_xg = 1.35
    season_pcs = pd.to_numeric(out["p_clean_sheet"], errors="coerce")
    fixture_rows = []
    horizon_pcs: dict[Any, float] = {}
    for fpl_id, row in out.iterrows():
        team = row["team_id"]
        own_def = team_lookup["team_defence"].get(team, np.nan) if pd.notna(team) else np.nan
        own_xga = float(-own_def) if pd.notna(own_def) else league_xg
        fxs = upc.loc[(upc["team_h"] == team) | (upc["team_a"] == team)] if pd.notna(team) else upc.iloc[0:0]
        by_event: dict[int, list] = {}
        for _, fr in fxs.iterrows():
            home = bool(fr["team_h"] == team)
            opp = fr["team_a"] if home else fr["team_h"]
            opp_att = team_lookup["team_attack"].get(opp, league_xg)
            opp_def_score = team_lookup["team_defence"].get(opp, -league_xg)
            leak = float(-opp_def_score) if pd.notna(opp_def_score) else league_xg
            att = float(opp_att) if pd.notna(opp_att) else league_xg
            ev = int(fr["event"]) if pd.notna(fr["event"]) else as_of_gw
            by_event.setdefault(ev, []).append(
                pack_fixture(
                    home=home,
                    opp_attack=att,
                    opp_defence_leak=leak,
                    own_xga=own_xga,
                    league_xg=league_xg,
                )
            )
        fx_json = json.dumps({str(k): v for k, v in by_event.items()})
        fixture_rows.append({"fpl_id": fpl_id, "fixtures_json": fx_json})
        fb = float(season_pcs.loc[fpl_id]) if fpl_id in season_pcs.index and pd.notna(season_pcs.loc[fpl_id]) else 0.22
        horizon_pcs[fpl_id] = horizon_p_cs(fx_json, settings.horizon_gws, fallback=fb)
    fxj = pd.DataFrame(fixture_rows).set_index("fpl_id")
    out["fixtures_json"] = fxj["fixtures_json"]
    out["p_clean_sheet"] = pd.Series(horizon_pcs).reindex(out.index).fillna(season_pcs).clip(0.02, 0.45)

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
