from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from fpl_model.config import Settings
from fpl_model.scoring.clean_sheet import (
    DEFAULT_LEAGUE_XG,
    dixon_coles_p_cs,
    expected_goals_against,
    expected_goals_for,
    isolate_event,
    iter_horizon_matches,
    parse_fixtures_json,
)

_SUMS_CACHE: dict[tuple[str, int, float], tuple] = {}
CACHE_VER = "v3"

GOAL_PTS = {"GK": 6.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0}
CS_PTS = {"GK": 4.0, "DEF": 4.0, "MID": 1.0, "FWD": 0.0}

# How much of the opponent leak feeds attack xPts (1 = full). CS/saves use the match itself.
TILT_BLEND = {"FWD": 1.00, "MID": 0.88, "DEF_CB": 0.42, "DEF_FB": 0.80, "GK": 0.10}


def _num(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").fillna(default)


def _match_is_home(m: dict) -> bool:
    v = m.get("home")
    try:
        return int(float(v)) == 1
    except (TypeError, ValueError):
        return False


def _blend_tilt(tilt: float, weight: float) -> float:
    return float(weight * tilt + (1.0 - weight) * 1.0)


def _match_components(m: dict, save_per_xg: float) -> tuple[float, float, float, float, float, float]:
    """Attack tilt, SP tilt, P(CS), expected saves, match xG, concede λ."""
    league = DEFAULT_LEAGUE_XG
    home = _match_is_home(m)
    own = float(m.get("own_xg") or league)
    leak = float(m.get("opp_defence_leak") or league)
    own_xga = float(m.get("own_xga") or league)
    opp = float(m.get("opp_attack") or league)
    match_xg = float(m["match_xg"]) if m.get("match_xg") is not None else expected_goals_for(own, leak, home=home, league_xg=league)
    match_xga = float(m["match_xga"]) if m.get("match_xga") is not None else expected_goals_against(own_xga, opp, home=home, league_xg=league)
    tilt = float(np.clip(match_xg / max(own, 0.55), 0.52, 1.48))
    tilt_sp = float(0.50 * tilt + 0.50)
    pcs = float(m["p_cs"]) if m.get("p_cs") is not None else dixon_coles_p_cs(match_xga, match_xg)
    saves = float(np.clip(save_per_xg * match_xga, 0.8, 6.5))
    return tilt, tilt_sp, pcs, saves, match_xg, match_xga


def _horizon_match_sums(
    fixtures_json: Any, horizon: int, save_per_xg: float
) -> tuple:
    """n, open tilts (fwd/mid/cb/fb/gk), sp tilts, P(CS), saves, mean tilt/pcs, mean match xG."""
    if isinstance(fixtures_json, str):
        fx_key = fixtures_json
    else:
        fx_key = repr(fixtures_json)
    cache_key = (CACHE_VER + fx_key, int(horizon), round(float(save_per_xg), 4))
    hit = _SUMS_CACHE.get(cache_key)
    if hit is not None:
        return hit
    matches = iter_horizon_matches(fixtures_json, horizon)
    empty = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.22, DEFAULT_LEAGUE_XG)
    if not matches:
        if len(_SUMS_CACHE) < 4000:
            _SUMS_CACHE[cache_key] = empty
        return empty
    tf = tm = tcb = tfb = tg = tsp = pcs = saves = mxg = 0.0
    for m in matches:
        t, t_sp, p, sav, mx, _ag = _match_components(m, save_per_xg)
        tf += _blend_tilt(t, TILT_BLEND["FWD"])
        tm += _blend_tilt(t, TILT_BLEND["MID"])
        tcb += _blend_tilt(t, TILT_BLEND["DEF_CB"])
        tfb += _blend_tilt(t, TILT_BLEND["DEF_FB"])
        tg += _blend_tilt(t, TILT_BLEND["GK"])
        tsp += t_sp
        pcs += p
        saves += sav
        mxg += mx
    n = float(len(matches))
    out = (n, tf, tm, tcb, tfb, tg, tsp, pcs, saves, tf / n, pcs / n, mxg / n)
    if len(_SUMS_CACHE) < 4000:
        _SUMS_CACHE[cache_key] = out
    return out


def _first_packed_event(fixtures_json: Any) -> int | None:
    by = parse_fixtures_json(fixtures_json)
    if not by:
        return None
    return min(int(k) for k in by.keys())


def compute_xpts_weeks(df: pd.DataFrame, weights: dict[str, Any], settings: Settings) -> pd.DataFrame:
    """Per-GW xPts for as-of … as-of+horizon-1 (blanks 0). Same rows as `df`."""
    hz = max(1, int(settings.horizon_gws))
    gw1 = replace(settings, horizon_gws=1)
    fx = df["fixtures_json"] if "fixtures_json" in df.columns else None
    as_of = None
    if fx is not None:
        for j in fx:
            ev = _first_packed_event(j)
            if ev is not None:
                as_of = ev if as_of is None else min(as_of, ev)
                break
    if as_of is None:
        return pd.DataFrame({0: compute_xpts(df, weights, gw1)}, index=df.index)
    cols: dict[int, pd.Series] = {}
    for i in range(hz):
        work = df
        if fx is not None:
            work = df.copy()
            ev = as_of + i
            work["fixtures_json"] = fx.map(lambda j, e=ev: isolate_event(j, e))
        if i == 0 and "availability" in work.columns:
            work["availability_horizon"] = work["availability"]
        elif i > 0 and "availability_later" in work.columns:
            work["availability_horizon"] = work["availability_later"]
        cols[i] = compute_xpts(work, weights, gw1)
    return pd.DataFrame(cols, index=df.index)


def compute_xpts(
    df: pd.DataFrame, weights: dict[str, Any], settings: Settings, *, return_parts: bool = False
) -> pd.Series | tuple[pd.Series, pd.DataFrame]:
    """Expected FPL points over the horizon from minutes, npxG/A, fixture P(CS), DefCon, saves.

    Attack, CS, saves, and bonus are summed per packed match (blanks add 0, DGW both).
    """
    cfg = weights.get("xpts") or {}
    horizon = int(settings.horizon_gws)
    l5_w = float(cfg.get("start_l5", 0.55))
    recency = float(cfg.get("recency", 0.45))
    # One hot L5 in a 2-match sample should not re-rate the player for every future GW.
    match_n = _num(df, "matches")
    recency = recency * (match_n / (match_n + 8.0)).clip(0.0, 1.0)
    assist_pts = float(cfg.get("assist", 3.0))
    defcon_mid = float(cfg.get("defcon_mid", 1.15))
    save_div = float(cfg.get("save_per_pt", 3.0))
    save_hist = float(cfg.get("save_hist", 0.45))
    save_per_xg = float(cfg.get("save_per_xg", 1.75))
    cs_scale = float(cfg.get("cs_scale", 0.65))
    p90_res = float(cfg.get("p90_residual", 0.10))
    p90_res_def = float(cfg.get("p90_residual_def", 0.06))
    defcon_def = float(cfg.get("defcon_def", 1.35))
    fwd_scale = float(cfg.get("fwd_attack_scale", 1.0))
    gk_attack = float(cfg.get("gk_attack_scale", 0.12))
    pen_xg = float(cfg.get("pen_xg90", 0.08))
    bonus_scale = float(cfg.get("bonus_scale", 0.55))
    bps_scale = float(cfg.get("bps_scale", 0.12))
    bonus_fx = float(cfg.get("bonus_fx", 0.22))
    bonus_cs = float(cfg.get("bonus_cs", 0.40))
    sp_goal_scale = float(cfg.get("sp_goal_scale", 1.12))
    yellow_pts = float(cfg.get("card_yellow", 1.0))

    start = _num(df, "start_pct")
    l5 = _num(df, "start_pct_l5")
    l5 = l5.where(l5.notna(), start)
    mins_pct = _num(df, "minutes_pct")
    if "minutes_pct_l5" in df.columns:
        mins_l5 = pd.to_numeric(df["minutes_pct_l5"], errors="coerce").fillna(mins_pct)
    else:
        mins_l5 = mins_pct
    if "availability_horizon" in df.columns:
        avail = _num(df, "availability_horizon", 1.0).clip(0.0, 1.0)
    else:
        avail = _num(df, "availability", 1.0).clip(0.0, 1.0)
    matches = _num(df, "matches")
    grace = getattr(settings, "prior_playing_grace_matches", 3)

    l5_w_eff = l5_w * (matches.clip(lower=0, upper=5) / 5.0)
    p_start = (1.0 - l5_w_eff) * start + l5_w_eff * l5
    p_start = ((p_start * avail).clip(0.0, 1.0)).astype(float)
    cold = (l5 <= 0.05) & (matches >= grace)
    p_start = p_start * np.where(np.asarray(cold), 0.25, 1.0)
    p_start = pd.Series(p_start, index=df.index)
    quirk = (mins_pct >= 0.70) & (l5 < 0.35) & ((matches < 4) | (start < 0.25))
    p_start = pd.Series(
        np.where(
            quirk.to_numpy(),
            np.maximum(p_start.to_numpy(), (mins_pct * avail).clip(0, 1).to_numpy()),
            p_start.to_numpy(),
        ),
        index=df.index,
    )

    rival_games = _num(df, "rival_games", 0.0)
    rot = (
        pd.to_numeric(df["rotation_start_pct"], errors="coerce")
        if "rotation_start_pct" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    use_rot = rot.notna() & (rival_games >= 3)
    p_start = pd.Series(
        np.where(
            use_rot.to_numpy(),
            np.minimum(p_start.to_numpy(), (0.35 + 0.65 * rot.clip(0, 1)).to_numpy()),
            p_start.to_numpy(),
        ),
        index=df.index,
    )
    slot = _num(df, "slot_security", 1.0).clip(0.40, 1.0)
    p_start = (p_start * slot).clip(0.0, 1.0)
    rest = _num(df, "rest_factor", 1.0).clip(0.75, 1.0)
    p_start = (p_start * rest).clip(0.0, 1.0)

    if "team_id" in df.columns and "position" in df.columns:
        gk = df["position"].eq("GK")
        if gk.any():
            tmax = df.loc[gk].groupby("team_id")["start_pct"].transform("max")
            share = (df.loc[gk, "start_pct"] / tmax.replace(0, np.nan)).fillna(0).clip(0, 1)
            p_start.loc[gk] = p_start.loc[gk] * share.to_numpy()

    l5_for_secure = pd.Series(np.maximum(l5.to_numpy(), mins_l5.to_numpy()), index=df.index)
    p_secure = pd.Series(
        np.minimum.reduce(
            [
                p_start.to_numpy(),
                (l5_for_secure * avail).clip(0, 1).to_numpy(),
            ]
        ),
        index=df.index,
    )

    fxcol = df["fixtures_json"] if "fixtures_json" in df.columns else pd.Series(["{}"] * len(df), index=df.index)
    bits = [_horizon_match_sums(r, horizon, save_per_xg) for r in fxcol]
    n_fx = pd.Series([b[0] for b in bits], index=df.index)
    tilt_fwd = pd.Series([b[1] for b in bits], index=df.index)
    tilt_mid = pd.Series([b[2] for b in bits], index=df.index)
    tilt_cb = pd.Series([b[3] for b in bits], index=df.index)
    tilt_fb = pd.Series([b[4] for b in bits], index=df.index)
    tilt_gk = pd.Series([b[5] for b in bits], index=df.index)
    sum_tilt_sp = pd.Series([b[6] for b in bits], index=df.index)
    sum_pcs = pd.Series([b[7] for b in bits], index=df.index)
    sum_saves_fx = pd.Series([b[8] for b in bits], index=df.index)
    mean_tilt = pd.Series([b[9] for b in bits], index=df.index)
    mean_pcs = pd.Series([b[10] for b in bits], index=df.index)
    mean_match_xg = pd.Series([b[11] for b in bits], index=df.index)

    npxg = _num(df, "npxg90").clip(lower=0)
    open_xg = (
        pd.to_numeric(df["open_npxg90"], errors="coerce")
        if "open_npxg90" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    sp_xg = (
        pd.to_numeric(df["sp_xg90"], errors="coerce")
        if "sp_xg90" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    has_split = open_xg.notna() & sp_xg.notna() & ((open_xg + sp_xg) > 0)
    open_use = npxg.where(~has_split, open_xg.fillna(0))
    sp_use = pd.Series(0.0, index=df.index).where(~has_split, sp_goal_scale * sp_xg.fillna(0))

    pos = df["position"].fillna("MID") if "position" in df.columns else pd.Series("MID", index=df.index)
    pen_flag = (_num(df, "pen_taker") >= 0.5) | (_num(df, "penalties_order", 99.0) == 1.0)
    pen_this = np.where(
        pos.isin(["FWD", "MID"]) & pen_flag,
        pen_xg * (mean_match_xg / DEFAULT_LEAGUE_XG).clip(0.55, 1.55),
        0.0,
    )

    npxa = _num(df, "npxa90").clip(lower=0)
    gi = npxg + npxa
    if "npxgi90_l5" in df.columns:
        gi5 = pd.to_numeric(df["npxgi90_l5"], errors="coerce").fillna(gi)
    else:
        gi5 = gi
    gi_ratio = (gi5 / gi.replace(0, np.nan)).fillna(1.0).clip(0.55, 1.7)
    scale = (1.0 - recency) + recency * gi_ratio

    mins_frac = (mins_pct / start.replace(0, np.nan)).fillna(0.85).clip(0.45, 1.0)
    p_60 = pd.Series(
        (p_start * ((mins_frac - 0.50) / 0.22).clip(0.0, 1.0)).clip(0.0, 1.0),
        index=df.index,
    )
    p_60 = pd.Series(np.minimum(p_60.to_numpy(), p_secure.to_numpy()), index=df.index)
    p_cameo = (p_start - p_60).clip(0.0, 1.0)
    appear_ev = 2.0 * p_60 + 1.0 * p_cameo

    g_pts = pos.map(GOAL_PTS).fillna(5.0)
    cs_pts = pos.map(CS_PTS).fillna(0.0)
    fb = _num(df, "def_fb_share", 0.45).clip(0.0, 1.0)
    pitch = _num(df, "pitch_attack", 50.0)
    mid_role = (0.72 + 0.36 * (pitch / 100.0)).clip(0.72, 1.08)
    role_mult = np.where(pos.eq("FWD"), fwd_scale, 1.0)
    role_mult = np.where(pos.eq("GK"), gk_attack, role_mult)
    role_mult = np.where(pos.eq("MID"), mid_role, role_mult)
    role_mult = np.where(pos.eq("DEF"), 0.55 + 0.55 * fb, role_mult)
    sum_tilt_open = np.select(
        [pos.eq("FWD"), pos.eq("MID"), pos.eq("GK"), pos.eq("DEF")],
        [tilt_fwd, tilt_mid, tilt_gk, (1.0 - fb) * tilt_cb + fb * tilt_fb],
        default=tilt_mid,
    )
    attack_open = (g_pts * open_use + assist_pts * npxa) * scale * mins_frac * role_mult * sum_tilt_open
    attack_sp = (g_pts * (sp_use + pen_this)) * scale * mins_frac * role_mult * sum_tilt_sp
    attack = attack_open + attack_sp

    cs_horizon = cs_pts * sum_pcs * cs_scale
    cs_horizon = cs_horizon * np.where(pos.eq("DEF"), 1.12 - 0.28 * fb, 1.0)
    gk_cs = float(cfg.get("gk_cs_scale", 1.18))
    cs_horizon = cs_horizon * np.where(pos.eq("GK"), gk_cs, 1.0)
    cs_horizon = cs_horizon * rest

    defcon = _num(df, "defcon_hit_rate").clip(0, 1)
    defcon_xp = np.where(pos.eq("MID"), defcon_mid * defcon, 0.0) + np.where(
        pos.eq("DEF"), defcon_def * defcon * (1.08 - 0.20 * fb), 0.0
    )

    saves_hist = _num(df, "saves90").clip(0, 8)
    hist_w = save_hist * (match_n / (match_n + 10.0)).clip(0.0, 1.0)
    saves_total = (hist_w * saves_hist * n_fx + (1.0 - hist_w) * sum_saves_fx).clip(0, 8 * n_fx)
    save_pts = np.where(pos.eq("GK"), p_start * saves_total / save_div, 0.0)

    p90 = _num(df, "points90_blend")
    if "points90_blend" not in df.columns:
        p90 = _num(df, "points90")
    residual_w = np.where(pos.isin(["DEF", "GK"]), p90_res_def, p90_res)
    bonus90 = _num(df, "bonus90").clip(0, 3)
    bps90 = _num(df, "bps90").clip(0, 80)
    bonus_hist = bonus_scale * bonus90 + bps_scale * (bps90 / 20.0).clip(0, 2.0)
    bonus_atk = 1.0 + bonus_fx * (mean_tilt - 1.0)
    bonus_cs_term = 1.0 + bonus_cs * (mean_pcs - 0.22)
    bonus_mult = np.where(pos.isin(["GK", "DEF"]), bonus_cs_term, bonus_atk)
    bonus_from_events = 0.18 * (attack / n_fx.replace(0, np.nan)).fillna(0) + 0.12 * (cs_horizon / n_fx.replace(0, np.nan)).fillna(0)
    bonus_pts = n_fx * (0.55 * bonus_hist * bonus_mult * mins_frac + 0.45 * bonus_from_events) * p_60

    yellow90 = _num(df, "yellow90").clip(0, 0.6)
    card_xp = yellow_pts * yellow90

    appear_pts = n_fx * appear_ev
    residual_pts = p_start * n_fx * residual_w * p90.clip(0, 10)
    card_pts = p_start * n_fx * card_xp * (-1.0)
    attack_pts = p_start * attack
    cs_pts_tot = p_60 * cs_horizon
    defcon_pts_tot = p_60 * n_fx * defcon_xp
    xp = (appear_pts + residual_pts + bonus_pts + card_pts + attack_pts + save_pts + cs_pts_tot + defcon_pts_tot).astype(float).clip(lower=0.0)
    if not return_parts:
        return xp
    parts = pd.DataFrame(
        {
            "appear": appear_pts,
            "attack": attack_pts,
            "cs": cs_pts_tot,
            "saves": save_pts,
            "defcon": defcon_pts_tot,
            "bonus": bonus_pts,
        },
        index=df.index,
    )
    return xp, parts
