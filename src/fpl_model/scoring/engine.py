from __future__ import annotations

import json
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from fpl_model.config import Settings, load_settings, load_weights
from fpl_model.db import connect
from fpl_model.features.stats import to_score_0_100
from fpl_model.scoring.availability import minutes_risk
from fpl_model.scoring.upside import compute_upside
from fpl_model.scoring.xpts import compute_xpts, compute_xpts_weeks

POS = ("GK", "DEF", "MID", "FWD")


def _apply_minutes_reliability(df: pd.DataFrame, idx: pd.Series, pos: str, settings: Settings) -> None:
    """Keep M honest for unused players and #2 keepers (prior-inflated start_pct)."""
    m = df.loc[idx, "M"].astype(float)
    sub = df.index[idx]
    if settings.dampen_unused_minutes:
        if "start_pct_l5" in df.columns:
            l5 = df.loc[idx, "start_pct_l5"].fillna(0)
        else:
            l5 = pd.Series(0.0, index=sub)
        if "matches" in df.columns:
            matches = df.loc[idx, "matches"].fillna(0)
        else:
            matches = pd.Series(0.0, index=sub)
        cold = (l5 <= 0.05) & (matches >= settings.prior_playing_grace_matches)
        m = m * np.where(cold.to_numpy(), 0.42, 1.0)
    if pos == "GK" and "team_id" in df.columns and "start_pct" in df.columns:
        tmax = df.loc[idx].groupby("team_id")["start_pct"].transform("max")
        share = (df.loc[idx, "start_pct"] / tmax.replace(0, np.nan)).fillna(0).clip(0, 1)
        m = m * (0.22 + 0.78 * share)
    df.loc[idx, "M"] = m.clip(0, 100)


def _weighted(row: pd.Series, parts: dict[str, float]) -> float:
    total_w = 0.0
    acc = 0.0
    for k, w in parts.items():
        if w == 0 or k not in row or pd.isna(row[k]):
            continue
        acc += w * float(row[k])
        total_w += w
    if total_w == 0:
        return 50.0
    return acc / total_w * (sum(parts.values()) / total_w) if False else acc / total_w


def _norm_group(df: pd.DataFrame, cols: list[str], invert: set[str] | None = None) -> pd.DataFrame:
    invert = invert or set()
    scored = df.copy()
    for pos in df["position"].dropna().unique():
        idx = df["position"] == pos
        for c in cols:
            if c not in df.columns:
                continue
            scored.loc[idx, f"{c}_s"] = to_score_0_100(df.loc[idx, c], invert=c in invert)
    for c in cols:
        if f"{c}_s" not in scored.columns:
            scored[f"{c}_s"] = 50.0
        scored[f"{c}_s"] = scored[f"{c}_s"].fillna(50.0)
    return scored


def _rescore_mid_subpos(df: pd.DataFrame, cols: list[str], *, att_cut: float = 62.0, min_n: int = 10) -> pd.DataFrame:
    """Percentile F/U leaves among attacking vs defensive mids separately."""
    if "position" not in df.columns or "pitch_attack" not in df.columns:
        return df
    mid = df["position"] == "MID"
    if int(mid.sum()) < min_n * 2:
        return df
    att = pd.to_numeric(df["pitch_attack"], errors="coerce").fillna(70.0)
    am = mid & (att >= att_cut)
    dm = mid & ~am
    if int(am.sum()) < min_n or int(dm.sum()) < min_n:
        return df
    for c in cols:
        raw = c[:-2] if c.endswith("_s") else c
        if raw not in df.columns:
            continue
        scored = f"{raw}_s"
        if scored not in df.columns:
            continue
        for mask in (am, dm):
            df.loc[mask, scored] = to_score_0_100(df.loc[mask, raw])
    return df


def horizon_weights(hz: list[float], n: int) -> np.ndarray:
    """YAML lists a 5-GW decay. Stretch/renormalize when the scoring horizon differs."""
    base = [float(x) for x in hz]
    if n <= 0:
        return np.array([], dtype=float)
    if len(base) == n:
        w = np.array(base, dtype=float)
    elif len(base) > n:
        w = np.array(base[:n], dtype=float)
    else:
        tail = list(base)
        ratio = 0.7
        if len(base) >= 2 and base[-2] > 0:
            ratio = float(np.clip(base[-1] / base[-2], 0.5, 0.95))
        while len(tail) < n:
            tail.append(tail[-1] * ratio)
        w = np.array(tail, dtype=float)
    s = float(w.sum())
    return w / s if s else w


def _fixture_score(fixtures_json: str, position: str, weights: dict, horizon: int) -> float:
    try:
        by_event = json.loads(fixtures_json) if isinstance(fixtures_json, str) else {}
    except json.JSONDecodeError:
        by_event = {}
    fw = weights["fixtures"]
    pos_w = fw.get(position, fw["MID"])
    hz = horizon_weights(fw["horizon"], horizon)
    scores = []
    events = sorted(int(k) for k in by_event.keys()) if by_event else []
    if not events:
        return 40.0  # blank-ish
    min_e = min(events)
    for i in range(horizon):
        matches = by_event.get(str(min_e + i), [])
        if not matches:
            scores.append(0.0)
            continue
        piece = 0.0
        for m in matches:
            leak = m.get("opp_defence_leak", 1.2)
            att = m.get("opp_attack", 1.2)
            home = m.get("home", 0.5)
            pcs = m.get("p_cs", 0.25)
            ctx = m.get("team_xg_context")
            if ctx is None:
                ctx = 50.0
            s = (
                pos_w["opp_defence"] * min(100.0, leak / 2.0 * 100)
                + pos_w["opp_attack"] * (100.0 - min(100.0, att / 2.0 * 100))
                + pos_w["home_away"] * (70.0 if home else 40.0)
                + pos_w["team_xg_context"] * float(np.clip(ctx, 5.0, 100.0))
                + pos_w["p_cs"] * min(100.0, pcs * 200)
            )
            piece += s
        scores.append(piece)
    return float(np.dot(hz, np.array(scores, dtype=float)))


def _leaf_col(name: str, df: pd.DataFrame) -> str | None:
    col = f"{name}_s"
    if col in df.columns:
        return col
    alt = {
        "npxgi90": "npxgi90_s",
        "npxgi90_l5": "npxgi90_l5_s",
        "npxg90": "npxg90_s",
        "npxa90": "npxa90_s",
        "shots_in_box90": "shots_in_box90_s",
        "fbref_shots90": "fbref_shots90_s",
        "key_passes90": "key_passes90_s",
        "defcon_hit_rate": "defcon_hit_rate_s",
        "fbref_cbit90": "fbref_cbit90_s",
        "xgchain90": "xgchain90_s",
        "xgbuildup90": "xgbuildup90_s",
        "open_npxg90": "open_npxg90_s",
        "sp_xg90": "sp_xg90_s",
        "chance_quality90": "chance_quality90_s",
        "p_clean_sheet": "p_clean_sheet_s",
        "team_defence": "team_defence_s",
        "saves90": "saves90_s",
        "fpl_output": "points90_s",
        "penalty_exposure": "pen_score_s" if "pen_score_s" in df.columns else None,
        "shot_stopping": "shot_stopping_s" if "shot_stopping_s" in df.columns else None,
    }.get(name, f"{name}_s")
    if alt is None or alt not in df.columns:
        return None
    return alt


def _underlying_score(df: pd.DataFrame, idx: pd.Series, leaf: dict[str, float]) -> pd.Series:
    acc = pd.Series(0.0, index=df.index[idx])
    tw = pd.Series(0.0, index=df.index[idx])
    for name, w in leaf.items():
        if w <= 0:
            continue
        col = _leaf_col(name, df)
        if col is None:
            continue
        s = df.loc[idx, col]
        present = s.notna()
        acc = acc + (w * s).where(present, 0.0)
        tw = tw + w * present.astype(float)
    return (acc / tw).where(tw > 0, 50.0)


def score_feature_frame(feats: pd.DataFrame, weights: dict[str, Any], settings: Settings) -> pd.DataFrame:
    df = feats.copy()
    u_cols = [
        "npxgi90",
        "npxgi90_l5",
        "npxg90",
        "npxa90",
        "shots_in_box90",
        "fbref_shots90",
        "key_passes90",
        "defcon_hit_rate",
        "fbref_cbit90",
        "xgchain90",
        "xgbuildup90",
        "open_npxg90",
        "sp_xg90",
        "chance_quality90",
        "p_clean_sheet",
        "team_defence",
        "saves90",
        "shot_stopping",
        "pen_score",
    ]
    f_cols = ["points90_blend", "points90", "goals90", "assists90", "bonus90", "bps90"]
    m_cols = ["start_pct", "minutes_pct", "start_pct_l5", "rotation_start_pct", "slot_security", "availability"]
    role_cols = ["pitch_attack", "pitch_central", "npxgi90", "threat90"]

    df = _norm_group(
        df,
        [c for c in u_cols + f_cols + m_cols + role_cols + ["team_attack", "price", "points_form", "team_tilt"] if c in df.columns],
        invert=set(),
    )
    # Keep FBref scores missing when the raw rate is missing (unmatched / no minutes)
    for raw in ("fbref_shots90", "fbref_cbit90", "npxgi90_l5"):
        scored = f"{raw}_s"
        if scored in df.columns and raw in df.columns:
            df.loc[df[raw].isna(), scored] = np.nan

    df = _rescore_mid_subpos(
        df,
        [
            "npxgi90",
            "npxgi90_l5",
            "npxg90",
            "npxa90",
            "shots_in_box90",
            "key_passes90",
            "xgchain90",
            "chance_quality90",
            "open_npxg90",
            "points90_blend",
            "points90",
            "goals90",
            "assists90",
            "bonus90",
            "bps90",
        ],
    )

    df["U"] = 50.0
    df["F"] = 50.0
    df["R"] = 50.0
    df["T"] = 50.0
    df["M"] = 50.0
    df["O"] = 50.0
    df["V"] = 50.0

    uw = weights["underlying"]
    fo = weights["fpl_output"]
    role = weights["role"]
    tm = weights["team_mix"]
    mw = weights["minutes"]

    for pos in POS:
        idx = df["position"] == pos
        if not idx.any():
            continue
        # Underlying — skip missing leaves per player (e.g. no FBref map) and renormalize
        leaf = uw[pos]
        u = _underlying_score(df, idx, leaf)
        if pos == "MID" and "MID_DM" in uw:
            u_dm = _underlying_score(df, idx, uw["MID_DM"])
            pitch = df.loc[idx, "pitch_attack"] if "pitch_attack" in df.columns else 70.0
            att_share = ((pd.Series(pitch, index=df.index[idx]).fillna(70.0) - 40.0) / 52.0).clip(0.0, 1.0)
            if "understat_matched" in df.columns:
                matched = df.loc[idx, "understat_matched"].fillna(0).astype(float) > 0
                att_share = att_share.where(matched, 1.0)
            u = (1.0 - att_share) * u_dm + att_share * u
        if pos == "DEF" and "DEF_CB" in uw and "DEF_FB" in uw:
            u_cb = _underlying_score(df, idx, uw["DEF_CB"])
            u_fb = _underlying_score(df, idx, uw["DEF_FB"])
            fb = df.loc[idx, "def_fb_share"] if "def_fb_share" in df.columns else 0.45
            fb = pd.Series(fb, index=df.index[idx]).fillna(0.45).clip(0.0, 1.0)
            u = (1.0 - fb) * u_cb + fb * u_fb
        df.loc[idx, "U"] = u

        f_acc = pd.Series(0.0, index=df.index[idx])
        fo_use = {("points90_blend" if k == "points90" and "points90_blend_s" in df.columns else k): w for k, w in fo.items()}
        for name, w in fo_use.items():
            col = f"{name}_s"
            if col not in df.columns:
                continue
            f_acc = f_acc + w * df.loc[idx, col].fillna(50.0)
        df.loc[idx, "F"] = f_acc

        att_inv = df.loc[idx, "npxgi90_s"] if "npxgi90_s" in df.columns else 50.0
        if "threat90_s" in df.columns:
            att_inv = 0.6 * pd.Series(att_inv, index=df.index[idx]).fillna(50) + 0.4 * df.loc[idx, "threat90_s"].fillna(50)
        pitch = df.loc[idx, "pitch_attack"] if "pitch_attack" in df.columns else 50.0
        central = df.loc[idx, "pitch_central"] if "pitch_central" in df.columns else 50.0
        pen = df.loc[idx, "pen_score"] if "pen_score" in df.columns else 38.0
        sp = df.loc[idx, "sp_score"] if "sp_score" in df.columns else 40.0
        if "sp_xg90_s" in df.columns:
            sp = 0.55 * pd.Series(sp, index=df.index[idx]).fillna(40) + 0.45 * df.loc[idx, "sp_xg90_s"].fillna(50)
        minutes_imp = df.loc[idx, "start_pct_s"]
        df.loc[idx, "R"] = (
            role["pitch_position"] * pd.Series(pitch, index=df.index[idx]).fillna(50)
            + role["attacking_involvement"] * pd.Series(att_inv, index=df.index[idx]).fillna(50)
            + role["set_pieces"] * pd.Series(sp, index=df.index[idx]).fillna(40)
            + role["penalties"] * pd.Series(pen, index=df.index[idx]).fillna(38)
            + role["advanced_central"] * pd.Series(central, index=df.index[idx]).fillna(50)
            + role["tactical_importance"] * minutes_imp.fillna(50)
        )

        mix = tm[pos]
        atk = df.loc[idx, "team_attack_s"] if "team_attack_s" in df.columns else 50
        de = df.loc[idx, "team_defence_s"] if "team_defence_s" in df.columns else 50
        tscore = mix["attack"] * atk + mix["defence"] * de
        if "team_tilt_s" in df.columns:
            tw = {"FWD": 0.14, "MID": 0.12, "DEF": 0.08, "GK": 0.04}.get(pos, 0.08)
            tilt = pd.Series(df.loc[idx, "team_tilt_s"], index=df.index[idx])
            tscore = (1.0 - tw) * tscore + tw * tilt.fillna(50)
        df.loc[idx, "T"] = tscore

        rot_s = df.loc[idx, "rotation_start_pct_s"] if "rotation_start_pct_s" in df.columns else df.loc[idx, "start_pct_l5_s"]
        slot = df.loc[idx, "slot_security"].fillna(1.0) * 100 if "slot_security" in df.columns else 100.0
        df.loc[idx, "M"] = (
            mw["start_pct"] * df.loc[idx, "start_pct_s"].fillna(50)
            + mw["minutes_pct"] * df.loc[idx, "minutes_pct_s"].fillna(50)
            + mw["rotation"] * (0.65 * pd.Series(rot_s, index=df.index[idx]).fillna(50) + 0.35 * pd.Series(slot, index=df.index[idx]).fillna(100))
            + mw["availability"] * (df.loc[idx, "availability"].fillna(1) * 100)
        ).clip(0, 100)
        if "rest_factor" in df.columns:
            df.loc[idx, "M"] = (df.loc[idx, "M"] * df.loc[idx, "rest_factor"].fillna(1.0).clip(0.75, 1.0)).clip(0, 100)
        _apply_minutes_reliability(df, idx, pos, settings)
        # News / FPL chance is a hard minutes cap, not a 15% leaf.
        if "availability" in df.columns:
            avail = pd.to_numeric(df.loc[idx, "availability"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
            df.loc[idx, "M"] = (df.loc[idx, "M"] * (0.18 + 0.82 * avail)).clip(0, 100)

    if "understat_matched" in df.columns:
        um = pd.to_numeric(df["understat_matched"], errors="coerce").fillna(0.0)
        df["U"] = np.where(um < 1.0, 0.62 * df["U"] + 0.38 * 50.0, df["U"])
    if "matches" in df.columns:
        n = pd.to_numeric(df["matches"], errors="coerce").fillna(0.0)
        thin = n < 6
        df.loc[thin, "U"] = 0.58 * df.loc[thin, "U"] + 0.42 * 50.0

    # Fixtures
    df["O"] = [
        _fixture_score(r.get("fixtures_json", "{}"), r.get("position") or "MID", weights, settings.horizon_gws)
        for _, r in df.iterrows()
    ]
    df["O"] = df.groupby("position")["O"].transform(lambda s: to_score_0_100(s))
    # Injured / banned players should not keep full fixture credit on the card.
    if "availability" in df.columns:
        av = pd.to_numeric(df["availability"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
        df["O"] = (df["O"] * (0.55 + 0.45 * av)).clip(0, 100)

    q = weights["quality"]
    df["Q"] = q["U"] * df["U"] + q["F"] * df["F"] + q["R"] * df["R"] + q["T"] * df["T"]
    q_pos = weights.get("quality_by_pos") or {}
    for pos, qw in q_pos.items():
        pidx = df["position"] == pos
        if not pidx.any():
            continue
        df.loc[pidx, "Q"] = (
            qw["U"] * df.loc[pidx, "U"]
            + qw["F"] * df.loc[pidx, "F"]
            + qw["R"] * df.loc[pidx, "R"]
            + qw["T"] * df.loc[pidx, "T"]
        )

    # Value: component xPts over the horizon (minutes × npxG/A × CS / DefCon / saves)
    xp, parts = compute_xpts(df, weights, settings, return_parts=True)
    df["xpts"] = xp
    df["xpts_parts"] = [
        {k: round(float(parts.loc[i, k]), 2) for k in parts.columns} for i in df.index
    ]
    weeks = compute_xpts_weeks(df, weights, settings)
    df["xpts_gws"] = weeks.apply(lambda r: [round(float(v), 3) for v in r.tolist()], axis=1)
    df["xpts_next"] = weeks.iloc[:, 0] if not weeks.empty else df["xpts"]
    df["xpts_per_m"] = df["xpts"] / df["price"].replace(0, np.nan)
    df["q_per_m"] = df["Q"] / df["price"].replace(0, np.nan)
    df = _norm_group(df, ["xpts_per_m", "q_per_m"], invert=set())
    # premium: high price vs position median is worse
    df["premium_raw"] = df["price"] / df.groupby("position")["price"].transform("median")
    df = _norm_group(df, ["premium_raw"], invert={"premium_raw"})
    vw = weights["value"]
    df["V"] = (
        vw["xpts_per_m"] * df["xpts_per_m_s"].fillna(50)
        + vw["quality_per_m"] * df["q_per_m_s"].fillna(50)
        + vw["premium"] * df["premium_raw_s"].fillna(50)
    )

    ow = weights["overall"]
    by_pos = weights.get("overall_by_pos") or {}
    df["overall"] = ow["Q"] * df["Q"] + ow["M"] * df["M"] + ow["O"] * df["O"] + ow["V"] * df["V"]
    for pos, pow_ in by_pos.items():
        pidx = df["position"] == pos
        if not pidx.any():
            continue
        df.loc[pidx, "overall"] = (
            pow_["Q"] * df.loc[pidx, "Q"]
            + pow_["M"] * df.loc[pidx, "M"]
            + pow_["O"] * df.loc[pidx, "O"]
            + pow_["V"] * df.loc[pidx, "V"]
        )

    df = compute_upside(df, weights)

    avail = (
        pd.to_numeric(df["availability"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
        if "availability" in df.columns
        else pd.Series(1.0, index=df.index)
    )
    # Soft-dampen Overall when news says they are unlikely to play this GW.
    low = avail < 0.95
    df.loc[low, "overall"] = (df.loc[low, "overall"] * (0.62 + 0.38 * avail.loc[low])).clip(0, 100)

    def action(r) -> str:
        a = 1.0
        if "availability" in r.index and pd.notna(r["availability"]):
            a = float(r["availability"])
        if a < 0.35:
            return "HOLD" if r["overall"] >= 55 else "AVOID"
        if r["overall"] >= 80 and (r["upside"] >= 55 or (r["M"] >= 80 and r["O"] >= 70)):
            return "STRONG BUY"
        if r["overall"] >= 70 and r["M"] >= 60:
            return "BUY"
        if r["overall"] >= 55:
            return "HOLD"
        return "AVOID"

    df["action"] = df.apply(action, axis=1)
    return df


def persist_scores(conn: sqlite3.Connection, season: str, as_of_gw: int, scored: pd.DataFrame) -> None:
    conn.execute("DELETE FROM score_player_asof WHERE season = ? AND as_of_gw = ?", (season, as_of_gw))
    rows = []
    for _, r in scored.iterrows():
        card = {
            "player_quality": round(float(r["Q"]), 1),
            "underlying": round(float(r["U"]), 1),
            "fpl_output": round(float(r["F"]), 1),
            "role": round(float(r["R"]), 1),
            "team": round(float(r["T"]), 1),
            "minutes": round(float(r["M"]), 1),
            "fixtures": round(float(r["O"]), 1),
            "value": round(float(r["V"]), 1),
            "overall": round(float(r["overall"]), 1),
            "upside": round(float(r["upside"]), 1),
            "upside_parts": {
                "finishing": round(float(r.get("up_finishing", 50)), 1),
                "creation": round(float(r.get("up_creation", 50)), 1),
                "minutes_trend": round(float(r.get("up_minutes_trend", 50)), 1),
                "team_trend": round(float(r.get("up_team_trend", 50)), 1),
                "role_change": round(float(r.get("up_role_change", 50)), 1),
                "price": round(float(r.get("up_price", 50)), 1),
                "ownership": round(float(r.get("up_ownership", 50)), 1),
            },
            "xpts": round(float(r["xpts"]), 2),
            "xpts_next": round(float(r["xpts_next"] if pd.notna(r.get("xpts_next")) else r["xpts"]), 2),
            "xpts_gws": list(r["xpts_gws"]) if isinstance(r.get("xpts_gws"), (list, tuple)) else None,
            "xpts_parts": r["xpts_parts"] if isinstance(r.get("xpts_parts"), dict) else None,
            "flags": {
                "understat": float(r["understat_matched"] or 0) >= 1
                if "understat_matched" in r.index
                else False,
                "thin_sample": float(r["matches"] or 0) < 6 if "matches" in r.index else False,
                "no_fbref": float(r["fbref_matched"] or 0) < 1 if "fbref_matched" in r.index else False,
            },
            "minutes_risk": minutes_risk(
                availability=r["availability"] if pd.notna(r.get("availability")) else 1.0,
                status=r.get("status"),
                news=r.get("news"),
                chance_this=r.get("chance_of_playing_this"),
                chance_next=r.get("chance_of_playing_next"),
                slot_security=r.get("slot_security"),
                pos_slot_rank=r.get("pos_slot_rank"),
                position=r.get("position"),
                start_pct_l5=r.get("start_pct_l5"),
                minutes=r.get("minutes"),
                starts=r.get("starts"),
                appearances=r.get("appearances"),
            ),
            "availability": round(float(r["availability"]), 3) if pd.notna(r.get("availability")) else 1.0,
            "availability_horizon": round(
                float(
                    r["availability_horizon"]
                    if pd.notna(r.get("availability_horizon"))
                    else (r["availability"] if pd.notna(r.get("availability")) else 1.0)
                ),
                3,
            ),
            "action": r["action"],
        }
        rows.append(
            (
                season,
                int(as_of_gw),
                int(r["fpl_id"]),
                r.get("position"),
                int(r["team_id"]) if pd.notna(r.get("team_id")) else None,
                r.get("web_name"),
                float(r["U"]),
                float(r["F"]),
                float(r["R"]),
                float(r["T"]),
                float(r["Q"]),
                float(r["M"]),
                float(r["O"]),
                float(r["V"]),
                float(r["overall"]),
                float(r["upside"]),
                float(r["xpts"]),
                r["action"],
                json.dumps(card),
            )
        )
    conn.executemany(
        """INSERT INTO score_player_asof(
            season, as_of_gw, fpl_id, position, team_id, web_name,
            U, F, R, T, Q, M, O, V, overall, upside, xpts, action, card_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()


def load_feats(
    conn: sqlite3.Connection,
    season: str,
    as_of_gw: int,
    fpl_ids: list[int] | None = None,
) -> pd.DataFrame:
    if fpl_ids:
        ids = [int(i) for i in fpl_ids]
        ph = ",".join("?" * len(ids))
        raw = pd.read_sql_query(
            f"SELECT * FROM feat_player_asof WHERE season = ? AND as_of_gw = ? AND fpl_id IN ({ph})",
            conn,
            params=(season, as_of_gw, *ids),
        )
    else:
        raw = pd.read_sql_query(
            "SELECT * FROM feat_player_asof WHERE season = ? AND as_of_gw = ?",
            conn,
            params=(season, as_of_gw),
        )
    if raw.empty:
        return raw
    parsed = raw["features_json"].map(lambda s: json.loads(s) if isinstance(s, str) else (s or {}))
    extra = pd.json_normalize(parsed)
    clash = [c for c in extra.columns if c in raw.columns]
    extra = extra.drop(columns=clash, errors="ignore")
    return pd.concat([raw.drop(columns=["features_json"]), extra], axis=1)


def persist_asof_snapshot(
    conn: sqlite3.Connection,
    season: str,
    as_of_gw: int,
    *,
    is_beta: bool,
    official_as_of: int | None,
) -> None:
    conn.execute(
        """
        INSERT INTO dim_asof_snapshot(season, as_of_gw, is_beta, official_as_of)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(season, as_of_gw) DO UPDATE SET
            is_beta = excluded.is_beta,
            official_as_of = excluded.official_as_of
        """,
        (season, int(as_of_gw), 1 if is_beta else 0, official_as_of),
    )
    conn.commit()


def score_asof(
    season: str,
    as_of_gw: int,
    settings: Settings | None = None,
    *,
    is_beta: bool = False,
    official_as_of: int | None = None,
) -> int:
    settings = settings or load_settings()
    weights = load_weights()
    conn = connect(settings)
    feats = load_feats(conn, season, as_of_gw)
    if feats.empty:
        conn.close()
        return 0
    scored = score_feature_frame(feats, weights, settings)
    persist_scores(conn, season, as_of_gw, scored)
    persist_asof_snapshot(
        conn,
        season,
        as_of_gw,
        is_beta=is_beta,
        official_as_of=official_as_of if official_as_of is not None else as_of_gw,
    )
    n = len(scored)
    conn.close()
    return n
