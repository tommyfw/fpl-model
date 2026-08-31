from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fpl_model.config import Settings
from fpl_model.scoring.clean_sheet import iter_horizon_matches

GOAL_PTS = {"GK": 6.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0}
CS_PTS = {"GK": 4.0, "DEF": 4.0, "MID": 1.0, "FWD": 0.0}


def _num(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").fillna(default)


def _horizon_matches(fixtures_json: Any, horizon: int) -> tuple[float, float, float, float]:
    """Count fixtures in the window; mean opponent leak/attack; sum of per-match P(CS)."""
    matches = iter_horizon_matches(fixtures_json, horizon)
    if not matches:
        return 0.0, 1.2, 1.2, 0.0
    leaks = [float(m.get("opp_defence_leak") or 1.2) for m in matches]
    atts = [float(m.get("opp_attack") or 1.2) for m in matches]
    pcs = [float(m.get("p_cs") if m.get("p_cs") is not None else 0.22) for m in matches]
    n = float(len(matches))
    return n, float(np.mean(leaks)), float(np.mean(atts)), float(np.sum(pcs))


def compute_xpts(df: pd.DataFrame, weights: dict[str, Any], settings: Settings) -> pd.Series:
    """Expected FPL points over the horizon from minutes, npxG/A, fixture P(CS), DefCon, saves.

    Fixture difficulty for attack is a small tilt. Clean sheets use per-match P(CS)
    (own xGA × opponent xG × H/A). Blanks/DGWs come from the fixture list.
    """
    cfg = weights.get("xpts") or {}
    horizon = int(settings.horizon_gws)
    l5_w = float(cfg.get("start_l5", 0.55))
    recency = float(cfg.get("recency", 0.45))
    fx_lo = float(cfg.get("fixture_lo", 0.85))
    fx_hi = float(cfg.get("fixture_hi", 1.22))
    assist_pts = float(cfg.get("assist", 3.0))
    defcon_pts = float(cfg.get("defcon", 2.0))
    save_div = float(cfg.get("save_per_pt", 3.0))
    cs_scale = float(cfg.get("cs_scale", 0.65))
    p90_res = float(cfg.get("p90_residual", 0.10))
    p90_res_def = float(cfg.get("p90_residual_def", 0.06))
    defcon_def = float(cfg.get("defcon_def", 1.5))
    fwd_scale = float(cfg.get("fwd_attack_scale", 1.10))
    bonus_scale = float(cfg.get("bonus_scale", 0.55))
    bps_scale = float(cfg.get("bps_scale", 0.12))

    start = _num(df, "start_pct")
    l5 = _num(df, "start_pct_l5")
    l5 = l5.where(l5.notna(), start)
    mins_pct = _num(df, "minutes_pct")
    if "minutes_pct_l5" in df.columns:
        mins_l5 = pd.to_numeric(df["minutes_pct_l5"], errors="coerce").fillna(mins_pct)
    else:
        mins_l5 = mins_pct
    avail = _num(df, "availability", 1.0).clip(0.0, 1.0)
    matches = _num(df, "matches")
    grace = getattr(settings, "prior_playing_grace_matches", 3)

    # Down-weight last-5 until a few GWs exist (one blank start flag must not zero a 5-GW forecast).
    l5_w_eff = l5_w * (matches.clip(lower=0, upper=5) / 5.0)
    p_start = (1.0 - l5_w_eff) * start + l5_w_eff * l5
    p_start = ((p_start * avail).clip(0.0, 1.0)).astype(float)
    cold = (l5 <= 0.05) & (matches >= grace)
    p_start = p_start * np.where(np.asarray(cold), 0.25, 1.0)
    p_start = pd.Series(p_start, index=df.index)
    # FPL can log 90 minutes with starts=0. Trust minutes when last-5 starts missed a 90.
    quirk = (mins_pct >= 0.70) & (l5 < 0.35) & ((matches < 4) | (start < 0.25))
    p_start = pd.Series(
        np.where(quirk.to_numpy(), np.maximum(p_start.to_numpy(), (mins_pct * avail).clip(0, 1).to_numpy()), p_start.to_numpy()),
        index=df.index,
    )

    rival_games = _num(df, "rival_games", 0.0)
    rot = pd.to_numeric(df["rotation_start_pct"], errors="coerce") if "rotation_start_pct" in df.columns else pd.Series(np.nan, index=df.index)
    use_rot = rot.notna() & (rival_games >= 3)
    p_start = pd.Series(
        np.where(use_rot.to_numpy(), np.minimum(p_start.to_numpy(), (0.35 + 0.65 * rot.clip(0, 1)).to_numpy()), p_start.to_numpy()),
        index=df.index,
    )
    slot = _num(df, "slot_security", 1.0).clip(0.40, 1.0)
    p_start = (p_start * slot).clip(0.0, 1.0)
    rest = _num(df, "rest_factor", 1.0).clip(0.82, 1.0)
    p_start = (p_start * rest).clip(0.0, 1.0)

    if "team_id" in df.columns and "position" in df.columns:
        gk = df["position"].eq("GK")
        if gk.any():
            tmax = df.loc[gk].groupby("team_id")["start_pct"].transform("max")
            share = (df.loc[gk, "start_pct"] / tmax.replace(0, np.nan)).fillna(0).clip(0, 1)
            p_start.loc[gk] = p_start.loc[gk] * share.to_numpy()

    # CS / DefCon need a 60+ start, not bench cameos. Use last-5 + rotation, not season minutes.
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
    bits = [_horizon_matches(r, horizon) for r in fxcol]
    n_fx = pd.Series([b[0] for b in bits], index=df.index)
    leak = pd.Series([b[1] for b in bits], index=df.index)
    sum_pcs = pd.Series([b[3] for b in bits], index=df.index)
    atk_fx = (leak / 1.2).clip(fx_lo, fx_hi)

    npxg = _num(df, "npxg90").clip(lower=0)
    npxa = _num(df, "npxa90").clip(lower=0)
    gi = npxg + npxa
    if "npxgi90_l5" in df.columns:
        gi5 = pd.to_numeric(df["npxgi90_l5"], errors="coerce").fillna(gi)
    else:
        gi5 = gi
    gi_ratio = (gi5 / gi.replace(0, np.nan)).fillna(1.0).clip(0.55, 1.7)
    scale = (1.0 - recency) + recency * gi_ratio

    mins_frac = (mins_pct / start.replace(0, np.nan)).fillna(0.85).clip(0.45, 1.0)
    appear = np.where(mins_frac >= 0.67, 2.0, 1.0)
    # FPL CS requires 60+ minutes; rotation pieces should not get a full 4-pt CS.
    cs_eligible = ((mins_frac - 0.50) / 0.17).clip(0.0, 1.0)

    pos = df["position"].fillna("MID") if "position" in df.columns else pd.Series("MID", index=df.index)
    g_pts = pos.map(GOAL_PTS).fillna(5.0)
    cs_pts = pos.map(CS_PTS).fillna(0.0)
    attack = (g_pts * npxg + assist_pts * npxa) * scale * mins_frac * atk_fx
    attack = attack * np.where(pos.eq("FWD"), fwd_scale, 1.0)
    fb = _num(df, "def_fb_share", 0.45).clip(0.0, 1.0)
    attack = attack * np.where(pos.eq("DEF"), 0.82 + 0.40 * fb, 1.0)

    cs_horizon = cs_pts * sum_pcs * cs_scale * cs_eligible
    cs_horizon = cs_horizon * np.where(pos.eq("DEF"), 1.12 - 0.28 * fb, 1.0)
    cs_horizon = cs_horizon * rest

    defcon = _num(df, "defcon_hit_rate").clip(0, 1)
    defcon_xp = np.where(pos.eq("MID"), defcon_pts * defcon, 0.0) + np.where(
        pos.eq("DEF"), defcon_def * defcon * (1.08 - 0.20 * fb), 0.0
    )

    saves = _num(df, "saves90").clip(0, 8)
    save_xp = np.where(pos.eq("GK"), saves / save_div, 0.0)

    p90 = _num(df, "points90_blend")
    if "points90_blend" not in df.columns:
        p90 = _num(df, "points90")
    residual_w = np.where(pos.isin(["DEF", "GK"]), p90_res_def, p90_res)
    bonus90 = _num(df, "bonus90").clip(0, 3)
    bps90 = _num(df, "bps90").clip(0, 80)
    bonus_pts = bonus_scale * bonus90 + bps_scale * (bps90 / 20.0).clip(0, 2.0)
    per_start = appear + attack + save_xp + residual_w * p90.clip(0, 10) + bonus_pts * mins_frac
    secure_pts = n_fx * defcon_xp + cs_horizon
    return (p_start * n_fx * per_start + p_secure * secure_pts).astype(float).clip(lower=0.0)
