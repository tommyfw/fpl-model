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
from fpl_model.scoring.availability import availability_horizon_mean
from fpl_model.scoring.clean_sheet import horizon_p_cs, pack_match_for_team
from fpl_model.ingest.priors import previous_season
from fpl_model.scoring.role import (
    def_fb_share,
    defcon_count,
    penalty_score,
    pitch_attack_score,
    pitch_central_score,
    set_piece_score,
)

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
        out = _attach_rest(out, fx, as_of_gw, pd.DataFrame(), cutoff)
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
        from fpl_model.ingest.understat import load_shot_splits

        splits = load_shot_splits(season, settings)
        if not splits.empty and "understat_id" in us.columns and "match_id" in us.columns:
            us = us.drop(columns=[c for c in ("open_npxg", "sp_xg") if c in us.columns], errors="ignore")
            us["match_id"] = pd.to_numeric(us["match_id"], errors="coerce")
            splits = splits.copy()
            splits["match_id"] = pd.to_numeric(splits["match_id"], errors="coerce")
            us = us.merge(splits, on=["understat_id", "match_id"], how="left")
        us["open_npxg"] = pd.to_numeric(us["open_npxg"], errors="coerce").fillna(0.0) if "open_npxg" in us.columns else 0.0
        us["sp_xg"] = pd.to_numeric(us["sp_xg"], errors="coerce").fillna(0.0) if "sp_xg" in us.columns else 0.0
        g = us.groupby("fpl_id", as_index=True).agg(
            us_minutes=("minutes", "sum"),
            us_npxg=("npxg", "sum"),
            us_npxa=("xa", "sum"),
            us_shots=("shots", "sum"),
            us_sib=("shots_in_box", "sum"),
            us_kp=("key_passes", "sum"),
            us_xgchain=("xgchain", "sum"),
            us_xgbuildup=("xgbuildup", "sum"),
            us_open=("open_npxg", "sum"),
            us_sp=("sp_xg", "sum"),
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
                ("xgbuildup90", "us_xgbuildup", None),
                ("open_npxg90", "us_open", None),
                ("sp_xg90", "us_sp", None),
            ]:
                raw = pd.Series(per90(g.loc[idx, src], minutes), index=idx).astype(float)
                if obs_dest:
                    out.loc[idx, obs_dest] = raw
                pos = out.loc[idx, "position"] if "position" in out.columns else None
                if pos is not None and pos.notna().any():
                    pos_mean = raw.groupby(pos).transform("mean")
                    pos_mean = pos_mean.fillna(float(raw.mean()) if raw.notna().any() else 0.0)
                else:
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
        # Always blend with prior-season xG (promoted sides → bottom-quartile PL).
        q = float(getattr(settings, "promoted_prior_quantile", 0.25))
        priors = load_team_priors(conn, season, promoted_quantile=q)
        atk, de = _blend_team_strength(conn, season, atk, de, n_obs, settings, priors=priors)
        min_venue = int(getattr(settings, "venue_prior_matches", 4))
        ut_v = ut if not ut.empty else pd.DataFrame()
        atk_h, atk_a, de_h, de_a = _venue_strength(
            ut_v,
            atk,
            de,
            min_venue,
            priors=priors,
        )
        n_home, n_away = _venue_match_counts(ut_v)
        out["team_attack"] = out["team_id"].map(atk).fillna(out.get("team_attack"))
        out["team_defence"] = out["team_id"].map(de).fillna(out.get("team_defence"))
        out["team_attack_home"] = out["team_id"].map(atk_h).fillna(out["team_attack"])
        out["team_attack_away"] = out["team_id"].map(atk_a).fillna(out["team_attack"])
        out["team_defence_home"] = out["team_id"].map(de_h).fillna(out["team_defence"])
        out["team_defence_away"] = out["team_id"].map(de_a).fillna(out["team_defence"])
        out["venue_n_home"] = out["team_id"].map(n_home).fillna(0.0)
        out["venue_n_away"] = out["team_id"].map(n_away).fillna(0.0)
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
    out = _attach_rest(out, fx, as_of_gw, us if not us.empty else pd.DataFrame(), cutoff)
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


def attach_horizon_fixtures(
    feats: pd.DataFrame,
    fx: pd.DataFrame,
    as_of_gw: int,
    horizon: int,
) -> pd.DataFrame:
    """Repack ``fixtures_json`` for T .. T+horizon-1 without changing stored 5-GW scores."""
    if feats.empty or fx is None or fx.empty or horizon < 1:
        return feats
    out = feats.copy()
    upc = upcoming_fixtures(fx, as_of_gw, int(horizon))
    team_cols = [
        c
        for c in (
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
        if c in out.columns
    ]
    if "team_id" not in out.columns or not team_cols:
        return out
    lookup = out.dropna(subset=["team_id"]).groupby("team_id")[team_cols].first()
    lookup.index = lookup.index.astype(int)
    atk = lookup["team_attack"].dropna() if "team_attack" in lookup.columns else pd.Series(dtype=float)
    league_xg = float(atk.mean()) if len(atk) else 1.35
    if not np.isfinite(league_xg) or league_xg <= 0.4:
        league_xg = 1.35
    packed: list[str] = []
    for _, row in out.iterrows():
        team = row.get("team_id")
        if pd.isna(team):
            packed.append("{}")
            continue
        team = int(team)
        fxs = upc.loc[(upc["team_h"] == team) | (upc["team_a"] == team)] if not upc.empty else upc
        by_event: dict[int, list] = {}
        for _, fr in fxs.iterrows():
            if pd.isna(fr.get("event")):
                continue
            home = bool(int(fr["team_h"]) == team)
            opp = fr["team_a"] if home else fr["team_h"]
            ev = int(fr["event"])
            by_event.setdefault(ev, []).append(
                pack_match_for_team(
                    home=home,
                    team_id=team,
                    opp_id=opp,
                    lookup=lookup,
                    league_xg=league_xg,
                    position=str(row.get("position") or "MID"),
                )
            )
        packed.append(json.dumps({str(k): v for k, v in by_event.items()}))
    out["fixtures_json"] = packed
    return out


def attach_horizon_fixtures(
    feats: pd.DataFrame,
    fx: pd.DataFrame,
    as_of_gw: int,
    horizon: int,
) -> pd.DataFrame:
    """Repack ``fixtures_json`` for T .. T+horizon-1 without changing stored 5-GW scores."""
    if feats.empty or fx is None or fx.empty or horizon < 1:
        return feats
    out = feats.copy()
    upc = upcoming_fixtures(fx, as_of_gw, int(horizon))
    team_cols = [
        c
        for c in (
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
        if c in out.columns
    ]
    if "team_id" not in out.columns or not team_cols:
        return out
    lookup = out.dropna(subset=["team_id"]).groupby("team_id")[team_cols].first()
    lookup.index = lookup.index.astype(int)
    atk = lookup["team_attack"].dropna() if "team_attack" in lookup.columns else pd.Series(dtype=float)
    league_xg = float(atk.mean()) if len(atk) else 1.35
    if not np.isfinite(league_xg) or league_xg <= 0.4:
        league_xg = 1.35
    packed: list[str] = []
    for _, row in out.iterrows():
        team = row.get("team_id")
        if pd.isna(team):
            packed.append("{}")
            continue
        team = int(team)
        fxs = upc.loc[(upc["team_h"] == team) | (upc["team_a"] == team)] if not upc.empty else upc
        by_event: dict[int, list] = {}
        for _, fr in fxs.iterrows():
            if pd.isna(fr.get("event")):
                continue
            home = bool(int(fr["team_h"]) == team)
            opp = fr["team_a"] if home else fr["team_h"]
            ev = int(fr["event"])
            by_event.setdefault(ev, []).append(
                pack_match_for_team(
                    home=home,
                    team_id=team,
                    opp_id=opp,
                    lookup=lookup,
                    league_xg=league_xg,
                    position=str(row.get("position") or "MID"),
                )
            )
        packed.append(json.dumps({str(k): v for k, v in by_event.items()}))
    out["fixtures_json"] = packed
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


# Pull venue rates toward this-season overall so n=1 home/away splits
# cannot dominate CS chips (last-season venue is only a light tilt).
VENUE_SEASON_MIX = 0.65


def _venue_match_counts(ut: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    empty = pd.Series(dtype=float)
    if ut is None or ut.empty or "team_id" not in ut.columns or "is_home" not in ut.columns:
        return empty, empty
    u = ut.dropna(subset=["team_id"]).copy()
    u["team_id"] = pd.to_numeric(u["team_id"], errors="coerce")
    u = u.dropna(subset=["team_id"])
    if u.empty:
        return empty, empty
    u["team_id"] = u["team_id"].astype(int)
    u["is_home"] = pd.to_numeric(u["is_home"], errors="coerce").fillna(0).astype(int)
    n_col = "match_id" if "match_id" in u.columns else "xg_for"
    g = u.groupby(["team_id", "is_home"], as_index=True).agg(
        n=(n_col, "nunique" if n_col == "match_id" else "count")
    )
    if g.empty:
        return empty, empty
    home = pd.Series(dtype=float)
    away = pd.Series(dtype=float)
    for (tid, is_home), row in g.iterrows():
        n = float(row["n"])
        if int(is_home) == 1:
            home.loc[int(tid)] = n
        else:
            away.loc[int(tid)] = n
    return home, away


def _venue_strength(
    ut: pd.DataFrame,
    season_atk: pd.Series,
    season_de: pd.Series,
    min_n: int,
    priors: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Home/away xG for and −xGA, shrunk toward this-season overall.

    Last-season venue rates only tilt the prior. Until a club has many
    home (or away) games this year, one freak venue result cannot dominate.
    """
    prior_n = max(int(min_n), 1)
    ids = season_atk.index.union(season_de.index)
    if priors is not None and not priors.empty:
        ids = ids.union(priors.index)
    g = pd.DataFrame()
    atk = pd.Series(dtype=float)
    de = pd.Series(dtype=float)
    if not ut.empty and "is_home" in ut.columns and "team_id" in ut.columns:
        u = ut.dropna(subset=["team_id"]).copy()
        u["team_id"] = pd.to_numeric(u["team_id"], errors="coerce")
        u = u.dropna(subset=["team_id"])
        u["team_id"] = u["team_id"].astype(int)
        u["is_home"] = pd.to_numeric(u["is_home"], errors="coerce").fillna(0).astype(int)
        n_col = "match_id" if "match_id" in u.columns else "xg_for"
        g = u.groupby(["team_id", "is_home"], as_index=True).agg(
            n=(n_col, "nunique" if n_col == "match_id" else "count"),
            xgf=("xg_for", "sum"),
            xga=("xg_against", "sum"),
        )
        if not g.empty:
            g["n"] = g["n"].clip(lower=1)
            atk = g["xgf"] / g["n"]
            de = -(g["xga"] / g["n"])

    def _prior(tid, col: str, season: pd.Series) -> float:
        if priors is not None and not priors.empty and tid in priors.index and col in priors.columns:
            v = priors.loc[tid, col]
            if pd.notna(v):
                return float(v)
        if tid in season.index:
            v = season.loc[tid]
            if pd.notna(v):
                return float(v)
        return float("nan")

    def _obs_n(tid, is_home: int) -> float:
        if g.empty:
            return 0.0
        try:
            if (tid, is_home) in g.index:
                return float(g.loc[(tid, is_home), "n"])
        except (KeyError, TypeError):
            return 0.0
        return 0.0

    def _blend(is_home: int, season: pd.Series, venue: pd.Series, prior_col: str) -> pd.Series:
        out = season.reindex(ids)
        for tid in ids:
            n = _obs_n(tid, is_home)
            prior = _prior(tid, prior_col, season)
            season_v = float(season.loc[tid]) if tid in season.index and pd.notna(season.loc[tid]) else float("nan")
            if np.isfinite(prior) and np.isfinite(season_v):
                base = VENUE_SEASON_MIX * season_v + (1.0 - VENUE_SEASON_MIX) * prior
            elif np.isfinite(season_v):
                base = season_v
            else:
                base = prior
            obs = float("nan")
            if n > 0:
                try:
                    obs = float(venue.loc[(tid, is_home)])
                except (KeyError, TypeError):
                    obs = float("nan")
            if np.isfinite(obs) and n > 0 and np.isfinite(base):
                w = n / (n + prior_n)
                out.loc[tid] = w * obs + (1.0 - w) * base
            elif np.isfinite(base):
                out.loc[tid] = base
            elif np.isfinite(obs):
                out.loc[tid] = obs
        return out

    atk_h = _blend(1, season_atk, atk, "prior_attack_home")
    atk_a = _blend(0, season_atk, atk, "prior_attack_away")
    de_h = _blend(1, season_de, de, "prior_defence_home")
    de_a = _blend(0, season_de, de, "prior_defence_away")
    return atk_h, atk_a, de_h, de_a


def _attach_rest(
    out: pd.DataFrame,
    fx: pd.DataFrame,
    as_of_gw: int,
    us: pd.DataFrame,
    cutoff,
) -> pd.DataFrame:
    """Rest until next GW and recent match density (PL + other comps in Understat)."""
    out = out.copy()
    out["rest_hours"] = np.nan
    out["matches_8d"] = 0.0
    out["rest_factor"] = 1.0
    fx = fx.copy()
    fx["ko"] = pd.to_datetime(fx.get("kickoff_time"), utc=True, errors="coerce")
    nxt = fx.loc[fx["event"] == as_of_gw]
    last = fx.loc[fx["event"] < as_of_gw]
    team_next = {}
    for _, r in nxt.iterrows():
        if pd.isna(r.get("ko")):
            continue
        team_next[int(r["team_h"])] = r["ko"]
        team_next[int(r["team_a"])] = r["ko"]
    team_last = {}
    if not last.empty:
        for tid, g in pd.concat(
            [
                last[["team_h", "ko"]].rename(columns={"team_h": "tid"}),
                last[["team_a", "ko"]].rename(columns={"team_a": "tid"}),
            ]
        ).dropna(subset=["ko"]).groupby("tid"):
            team_last[int(tid)] = g["ko"].max()
    cut_ts = pd.Timestamp(cutoff)
    if getattr(cut_ts, "tzinfo", None) is None:
        cut_ts = cut_ts.tz_localize("UTC")
    else:
        cut_ts = cut_ts.tz_convert("UTC")
    recent_counts = pd.Series(0.0, index=out.index)
    last_us = pd.Series(pd.NaT, index=out.index)
    if not us.empty and "date" in us.columns:
        usd = us.copy()
        usd["date"] = pd.to_datetime(usd["date"], utc=True, errors="coerce")
        lo = cut_ts - pd.Timedelta(days=8)
        recent = usd.loc[(usd["date"] >= lo) & (usd["date"] < cut_ts)]
        if not recent.empty:
            recent_counts = recent.groupby("fpl_id")["date"].nunique().reindex(out.index).fillna(0)
        last_us = usd.groupby("fpl_id")["date"].max().reindex(out.index)
    for fid, row in out.iterrows():
        tid = row.get("team_id")
        nxt_ko = team_next.get(int(tid)) if pd.notna(tid) else None
        last_ko = team_last.get(int(tid)) if pd.notna(tid) else None
        lu = last_us.loc[fid] if fid in last_us.index else pd.NaT
        last_any = last_ko
        if pd.notna(lu):
            if last_any is None or pd.isna(last_any) or lu > last_any:
                last_any = lu
        hours = np.nan
        if nxt_ko is not None and pd.notna(last_any):
            hours = (nxt_ko - last_any).total_seconds() / 3600.0
        out.at[fid, "rest_hours"] = hours
        n8 = float(recent_counts.loc[fid]) if fid in recent_counts.index else 0.0
        out.at[fid, "matches_8d"] = n8
        rf = 1.0
        if np.isfinite(hours):
            rf = float(np.clip(hours / 96.0, 0.82, 1.0))
        if n8 >= 2:
            rf = min(rf, 0.90)
        if n8 >= 3:
            rf = min(rf, 0.82)
        if np.isfinite(hours) and hours < 72:
            rf = min(rf, 0.86)
        out.at[fid, "rest_factor"] = float(np.clip(rf, 0.75, 1.0))
    return out


def _blend_team_strength(
    conn: sqlite3.Connection,
    season: str,
    atk: pd.Series,
    de: pd.Series,
    n: pd.Series,
    settings: Settings,
    priors: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Shrink observed team xG rates toward prior-season (promoted → bottom-quartile PL)."""
    if priors is None:
        q = float(getattr(settings, "promoted_prior_quantile", 0.25))
        priors = load_team_priors(conn, season, promoted_quantile=q)
    if priors is None or priors.empty:
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


def _defcon_season_priors(
    conn: sqlite3.Connection,
    season: str,
    thresholds: dict,
    index: pd.Index,
) -> pd.Series:
    """Last-season DefCon hit rate keyed by this season's fpl_id (via FPL code)."""
    prev = previous_season(season)
    codes = pd.read_sql_query(
        "SELECT fpl_id, code FROM dim_player WHERE season = ? AND code IS NOT NULL",
        conn,
        params=(season,),
    )
    prev_players = pd.read_sql_query(
        "SELECT fpl_id, code, position FROM dim_player WHERE season = ? AND code IS NOT NULL",
        conn,
        params=(prev,),
    )
    if codes.empty or prev_players.empty:
        return pd.Series(np.nan, index=index)
    try:
        gw = pd.read_sql_query(
            """
            SELECT fpl_id, position, starts, clearances_blocks_interceptions, tackles, recoveries,
                   defensive_contribution
            FROM fact_fpl_player_gw
            WHERE season = ? AND starts >= 1
            """,
            conn,
            params=(prev,),
        )
    except Exception:
        return pd.Series(np.nan, index=index)
    if gw.empty:
        return pd.Series(np.nan, index=index)
    gw = gw.merge(prev_players[["fpl_id", "code"]], on="fpl_id", how="left")
    gw = gw.dropna(subset=["code"])
    if "position" not in gw.columns or gw["position"].isna().all():
        gw["position"] = gw["fpl_id"].map(prev_players.set_index("fpl_id")["position"])
    gw["defcon_hit"] = _defcon_hits(gw, thresholds)
    by_code = gw.groupby("code")["defcon_hit"].mean()
    link = codes.set_index("fpl_id")["code"].map(by_code)
    return link.reindex(index)


def _blend_defcon_hit_rate(
    out: pd.DataFrame,
    n_starts: pd.Series,
    *,
    season_prior: pd.Series,
    prior_n: int,
    thresholds: dict,
) -> pd.Series:
    """Shrink observed DefCon hit rate toward last-season / FBref CBIT prior."""
    obs = pd.to_numeric(out["defcon_hit_rate"], errors="coerce").fillna(0.0)
    n = n_starts.reindex(out.index).fillna(0.0).astype(float)
    prior = season_prior.reindex(out.index)
    # FBref CBIT/90 → rough P(hit) when last-season rate missing
    if "fbref_cbit90" in out.columns:
        cbit = pd.to_numeric(out["fbref_cbit90"], errors="coerce")
        thr = out["position"].map(lambda p: thresholds.get(p) if p in thresholds else None)
        thr_n = pd.to_numeric(thr, errors="coerce")
        fb = (cbit / (thr_n + 2.0)).clip(0.05, 0.85)
        prior = prior.where(prior.notna(), fb)
    prior = prior.fillna(0.0).clip(0.0, 1.0)
    pn = max(int(prior_n), 1)
    w = n / (n + pn)
    # No prior signal and no starts → stay at observed 0
    has_prior = season_prior.reindex(out.index).notna()
    if "fbref_cbit90" in out.columns:
        has_prior = has_prior | pd.to_numeric(out["fbref_cbit90"], errors="coerce").notna()
    blended = w * obs + (1.0 - w) * prior
    return pd.Series(np.where(has_prior | (n > 0), blended, obs), index=out.index).clip(0.0, 1.0)


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
        c
        for c in [
            "fpl_id",
            "position",
            "team_id",
            "web_name",
            "status",
            "chance_of_playing_next",
            "chance_of_playing_this",
            "news",
            "penalties_order",
            "corners_order",
            "direct_fk_order",
        ]
        if c in players.columns
    ]
    if "fpl_id" not in pcols:
        pcols = ["fpl_id"] + pcols
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
        # FPL live often logs 90 minutes with starts=0; treat 60+ mins as a start.
        past["minutes"] = pd.to_numeric(past["minutes"], errors="coerce").fillna(0)
        past["starts"] = np.maximum(
            pd.to_numeric(past["starts"], errors="coerce").fillna(0),
            (past["minutes"] >= 60).astype(int),
        )
        if "yellow_cards" not in past.columns:
            past["yellow_cards"] = 0
        if "red_cards" not in past.columns:
            past["red_cards"] = 0
        past["yellow_cards"] = pd.to_numeric(past["yellow_cards"], errors="coerce").fillna(0)
        past["red_cards"] = pd.to_numeric(past["red_cards"], errors="coerce").fillna(0)
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
            yellows=("yellow_cards", "sum"),
            reds=("red_cards", "sum"),
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
        if "yellows" not in out.columns:
            out["yellows"] = 0
        if "reds" not in out.columns:
            out["reds"] = 0
        for c in [
            "position",
            "team_id",
            "web_name",
            "status",
            "chance_of_playing_next",
            "chance_of_playing_this",
            "news",
            "penalties_order",
            "corners_order",
            "direct_fk_order",
        ]:
            if c in meta.columns:
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
            "yellows",
            "reds",
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
            "yellows",
            "reds",
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
    out.loc[low_sample, "rotation_start_pct"] = np.nan
    out["rotation_start_pct"] = out["rotation_start_pct"].fillna(out["start_pct"]).clip(0.0, 1.0)
    out["slot_security"] = pd.to_numeric(out.get("slot_security"), errors="coerce").fillna(1.0).clip(0.4, 1.0)
    out["pos_slot_rank"] = pd.to_numeric(out.get("pos_slot_rank"), errors="coerce").fillna(1.0)
    out["rival_games"] = pd.to_numeric(out.get("rival_games"), errors="coerce").fillna(0)

    player_priors = load_player_priors(conn, season)
    prior = settings.shrink_prior_minutes
    promo_scale = float(getattr(settings, "promoted_player_prior_scale", 0.78))
    promo_teams: set = set()
    try:
        tp = load_team_priors(conn, season, float(getattr(settings, "promoted_prior_quantile", 0.25)))
        if not tp.empty and "is_promoted" in tp.columns:
            promo_teams = set(tp.index[tp["is_promoted"] == 1].tolist())
    except Exception:
        promo_teams = set()
    no_prior = ~out.index.isin(player_priors.index) if not player_priors.empty else pd.Series(True, index=out.index)
    on_promo = out["team_id"].isin(promo_teams) if promo_teams else pd.Series(False, index=out.index)
    weak_prior = no_prior & on_promo
    attack_prior_cols = {
        "npxg90",
        "npxa90",
        "npxgi90",
        "points90",
        "goals90",
        "assists90",
        "threat90",
        "creativity90",
    }
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
        ("yellow90", "yellows"),
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
        if col in attack_prior_cols:
            blend = pd.Series(blend, index=out.index).where(~weak_prior, pos_mean * promo_scale)
        n_m = pd.to_numeric(out["matches"], errors="coerce").fillna(0) if "matches" in out.columns else 0
        extra = pd.Series(0.0, index=out.index)
        if col in attack_prior_cols:
            extra = np.where((n_m < 8) & out["position"].isin(["DEF", "GK"]), 500.0, 0.0)
            extra = extra + np.where((n_m < 8) & out["position"].isin(["MID", "FWD"]), 300.0, 0.0)
        elif col == "saves90":
            extra = np.where((n_m < 8) & out["position"].eq("GK"), 500.0, 0.0)
        prior_m = pd.Series(float(prior), index=out.index) + extra
        out[col] = shrink_to_mean(
            raw, out["minutes"], blend, prior_m, max_prior_weight=settings.max_prior_weight
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

    n_defcon_starts = pd.Series(0.0, index=out.index)
    if not past.empty:
        past = past.copy()
        past["defcon_hit"] = _defcon_hits(past, settings.defcon_threshold)
        started = past.loc[past["starts"] >= 1]
        hit_rate = started.groupby("fpl_id")["defcon_hit"].mean()
        n_defcon_starts = started.groupby("fpl_id").size().astype(float)
        out["defcon_hit_rate"] = hit_rate.reindex(out.index).fillna(0.0)
    else:
        out["defcon_hit_rate"] = 0.0

    # proxies until Understat overlays
    out["shots_in_box90"] = out["threat90"] / 10.0
    out["key_passes90"] = out["creativity90"] / 10.0
    out["chance_quality90"] = out["npxg90"]
    out["xgchain90"] = out["npxgi90"]
    out["xgbuildup90"] = out["npxgi90"]
    out["open_npxg90"] = out["npxg90"]
    out["sp_xg90"] = 0.0
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
    season_defcon = _defcon_season_priors(conn, season, settings.defcon_threshold, out.index)
    out["defcon_hit_rate"] = _blend_defcon_hit_rate(
        out,
        n_defcon_starts,
        season_prior=season_defcon,
        prior_n=int(getattr(settings, "defcon_prior_matches", 6)),
        thresholds=settings.defcon_threshold,
    )
    if "rest_factor" not in out.columns:
        out["rest_factor"] = 1.0
    if "gc" not in out.columns:
        out["gc"] = 0.0
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
    last_red = pd.Series(False, index=out.index)
    if not past_sorted.empty and "red_cards" in past_sorted.columns:
        last_gw = past_sorted.sort_values(["fpl_id", "round"]).groupby("fpl_id").tail(1)
        last_red = (
            (pd.to_numeric(last_gw.set_index("fpl_id")["red_cards"], errors="coerce").fillna(0) >= 1)
            .reindex(out.index)
            .fillna(False)
        )
    out["last_red"] = last_red.astype(bool)
    if "yellows" not in out.columns:
        out["yellows"] = 0
    if "yellow90" not in out.columns:
        out["yellow90"] = 0.0
    avail_this = []
    avail_later = []
    avail_hz = []
    for st, ct, cn, nw, lr in zip(
        out["status"] if "status" in out.columns else [None] * len(out),
        out["chance_of_playing_this"] if "chance_of_playing_this" in out.columns else [None] * len(out),
        out["chance_of_playing_next"] if "chance_of_playing_next" in out.columns else [None] * len(out),
        out["news"] if "news" in out.columns else [None] * len(out),
        out["last_red"],
    ):
        a0, a1, am = availability_horizon_mean(
            st, chance_this=ct, chance_next=cn, news=nw, horizon=settings.horizon_gws, last_red=lr
        )
        avail_this.append(a0)
        avail_later.append(a1)
        avail_hz.append(am)
    out["availability"] = avail_this
    out["availability_later"] = avail_later
    out["availability_horizon"] = avail_hz
    out["pen_taker"] = (out["penalties_order"] == 1).astype(float)
    out["set_piece_taker"] = (
        (out["corners_order"].fillna(99) <= 2) | (out["direct_fk_order"].fillna(99) <= 2)
    ).astype(float)
    if "us_position" not in out.columns:
        out["us_position"] = None
    out["pitch_attack"] = out["us_position"].map(lambda p: pitch_attack_score(p))
    out["pitch_central"] = out["us_position"].map(lambda p: pitch_central_score(p))
    out["def_fb_share"] = [
        def_fb_share(p, a) for p, a in zip(out["us_position"], out["pitch_attack"])
    ]
    gc90 = pd.Series(per90(out["gc"] if "gc" in out.columns else 0, out["minutes"]), index=out.index)
    out["gc90"] = gc90
    out["shot_stopping"] = pd.to_numeric(out.get("xgc90"), errors="coerce") - pd.to_numeric(out["gc90"], errors="coerce")
    if "psxg_net90" in out.columns:
        gk = out["position"].eq("GK") & out["psxg_net90"].notna()
        out.loc[gk, "shot_stopping"] = pd.to_numeric(out.loc[gk, "psxg_net90"], errors="coerce")
    # Early-season GK shot-stopping is noisy — shrink toward 0 (league-average stop) until minutes accrue.
    gk_mask = out["position"].eq("GK")
    if gk_mask.any():
        ss = pd.to_numeric(out.loc[gk_mask, "shot_stopping"], errors="coerce")
        gk_min = pd.to_numeric(out.loc[gk_mask, "minutes"], errors="coerce").fillna(0.0)
        prior_m = float(getattr(settings, "shrink_prior_minutes", 450))
        w = (gk_min / (gk_min + prior_m)).clip(0.0, 1.0)
        out.loc[gk_mask, "shot_stopping"] = (w * ss.fillna(0.0)).astype(float)
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
    for c, src in (
        ("team_attack_home", "team_attack"),
        ("team_attack_away", "team_attack"),
        ("team_defence_home", "team_defence"),
        ("team_defence_away", "team_defence"),
    ):
        if c not in out.columns:
            out[c] = out.get(src)
    for c in ("venue_n_home", "venue_n_away"):
        if c not in out.columns:
            out[c] = 0.0
    team_cols = [
        "team_attack",
        "team_defence",
        "team_attack_home",
        "team_attack_away",
        "team_defence_home",
        "team_defence_away",
        "venue_n_home",
        "venue_n_away",
    ]
    if "team_tilt" in out.columns:
        team_cols.append("team_tilt")
    team_lookup = out.groupby("team_id")[team_cols].first()
    team_lookup = team_lookup[pd.notna(team_lookup.index)].copy()
    team_lookup.index = team_lookup.index.astype(int)
    atk_mean = team_lookup["team_attack"].dropna()
    league_xg = float(atk_mean.mean()) if len(atk_mean) else 1.35
    if not np.isfinite(league_xg) or league_xg <= 0.4:
        league_xg = 1.35
    season_pcs = pd.to_numeric(out["p_clean_sheet"], errors="coerce")
    fixture_rows = []
    horizon_pcs: dict[Any, float] = {}
    for fpl_id, row in out.iterrows():
        team = row["team_id"]
        fxs = upc.loc[(upc["team_h"] == team) | (upc["team_a"] == team)] if pd.notna(team) else upc.iloc[0:0]
        by_event: dict[int, list] = {}
        for _, fr in fxs.iterrows():
            home = bool(fr["team_h"] == team)
            opp = fr["team_a"] if home else fr["team_h"]
            ev = int(fr["event"]) if pd.notna(fr["event"]) else as_of_gw
            by_event.setdefault(ev, []).append(
                pack_match_for_team(
                    home=home,
                    team_id=team,
                    opp_id=opp,
                    lookup=team_lookup,
                    league_xg=league_xg,
                    position=str(row.get("position") or "MID"),
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
