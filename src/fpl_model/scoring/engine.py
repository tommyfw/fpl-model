from __future__ import annotations

import json
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from fpl_model.config import Settings, load_settings, load_weights
from fpl_model.db import connect
from fpl_model.features.stats import to_score_0_100

POS = ("GK", "DEF", "MID", "FWD")


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


def _fixture_score(fixtures_json: str, position: str, weights: dict, horizon: int) -> float:
    try:
        by_event = json.loads(fixtures_json) if isinstance(fixtures_json, str) else {}
    except json.JSONDecodeError:
        by_event = {}
    fw = weights["fixtures"]
    pos_w = fw.get(position, fw["MID"])
    hz = fw["horizon"]
    scores = []
    # infer as_of from keys? caller should pass offset. Keys are event numbers.
    events = sorted(int(k) for k in by_event.keys()) if by_event else []
    if not events:
        return 40.0  # blank-ish
    min_e = min(events)
    for i in range(horizon):
        ev = str(min_e + i)
        # also try sequential from min
        matches = by_event.get(str(min_e + i), by_event.get(ev, []))
        if not matches:
            scores.append(0.0)
            continue
        piece = 0.0
        for m in matches:
            leak = m.get("opp_defence_leak", 1.2)
            att = m.get("opp_attack", 1.2)
            home = m.get("home", 0.5)
            pcs = m.get("p_cs", 0.25)
            # raw then will percentile globally later; here 0-100-ish heuristic
            s = (
                pos_w["opp_defence"] * min(100.0, leak / 2.0 * 100)
                + pos_w["opp_attack"] * (100.0 - min(100.0, att / 2.0 * 100))
                + pos_w["home_away"] * (70.0 if home else 40.0)
                + pos_w["team_xg_context"] * 50.0
                + pos_w["p_cs"] * min(100.0, pcs * 200)
            )
            piece += s
        scores.append(piece)
    # pad
    while len(scores) < len(hz):
        scores.append(0.0)
    return float(np.dot(hz[:horizon], scores[:horizon]))


def score_feature_frame(feats: pd.DataFrame, weights: dict[str, Any], settings: Settings) -> pd.DataFrame:
    df = feats.copy()
    u_cols = [
        "npxgi90",
        "npxg90",
        "npxa90",
        "shots_in_box90",
        "key_passes90",
        "defcon_hit_rate",
        "xgchain90",
        "chance_quality90",
        "p_clean_sheet",
        "team_defence",
        "saves90",
    ]
    f_cols = ["points90", "goals90", "assists90", "bonus90", "bps90"]
    m_cols = ["start_pct", "minutes_pct", "start_pct_l5", "availability"]
    invert = {"team_defence"}  # already negative ga; higher team_defence (less ga) is better — we stored -ga90 so don't invert

    df = _norm_group(df, [c for c in u_cols + f_cols + m_cols + ["team_attack", "price", "points_form"] if c in df.columns], invert=set())

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
        # Underlying
        leaf = uw[pos]
        acc = pd.Series(0.0, index=df.index[idx])
        tw = 0.0
        for name, w in leaf.items():
            col = f"{name}_s"
            if w <= 0:
                continue
            if col not in df.columns:
                # map aliases
                alt = {
                    "npxgi90": "npxgi90_s",
                    "npxg90": "npxg90_s",
                    "npxa90": "npxa90_s",
                    "shots_in_box90": "shots_in_box90_s",
                    "key_passes90": "key_passes90_s",
                    "defcon_hit_rate": "defcon_hit_rate_s",
                    "xgchain90": "xgchain90_s",
                    "chance_quality90": "chance_quality90_s",
                    "p_clean_sheet": "p_clean_sheet_s",
                    "team_defence": "team_defence_s",
                    "saves90": "saves90_s",
                    "fpl_output": "points90_s",
                    "penalty_exposure": None,
                    "shot_stopping": "xgc90_s" if "xgc90_s" in df.columns else None,
                }.get(name, f"{name}_s")
                if alt is None or alt not in df.columns:
                    continue
                col = alt
            acc = acc + w * df.loc[idx, col].fillna(50.0)
            tw += w
        df.loc[idx, "U"] = (acc / tw) if tw else 50.0

        f_acc = pd.Series(0.0, index=df.index[idx])
        for name, w in fo.items():
            f_acc = f_acc + w * df.loc[idx, f"{name}_s"].fillna(50.0)
        df.loc[idx, "F"] = f_acc

        # Role: mix of proxies
        pen = df.loc[idx, "pen_taker"].fillna(0) * 100
        sp = df.loc[idx, "set_piece_taker"].fillna(0) * 100
        att_inv = df.loc[idx, "threat90_s"] if "threat90_s" in df.columns else df.loc[idx, "npxgi90_s"]
        minutes_imp = df.loc[idx, "start_pct_s"]
        df.loc[idx, "R"] = (
            role["pitch_position"] * 50.0
            + role["attacking_involvement"] * att_inv.fillna(50)
            + role["set_pieces"] * (50 + 0.5 * sp)
            + role["penalties"] * (40 + 0.6 * pen)
            + role["advanced_central"] * att_inv.fillna(50)
            + role["tactical_importance"] * minutes_imp.fillna(50)
        )

        mix = tm[pos]
        atk = df.loc[idx, "team_attack_s"] if "team_attack_s" in df.columns else 50
        de = df.loc[idx, "team_defence_s"] if "team_defence_s" in df.columns else 50
        df.loc[idx, "T"] = mix["attack"] * atk + mix["defence"] * de

        df.loc[idx, "M"] = (
            mw["start_pct"] * df.loc[idx, "start_pct_s"].fillna(50)
            + mw["minutes_pct"] * df.loc[idx, "minutes_pct_s"].fillna(50)
            + mw["rotation"] * df.loc[idx, "start_pct_l5_s"].fillna(50)
            + mw["availability"] * (df.loc[idx, "availability"].fillna(1) * 100)
        ).clip(0, 100)

    # Fixtures
    df["O"] = [
        _fixture_score(r.get("fixtures_json", "{}"), r.get("position") or "MID", weights, settings.horizon_gws)
        for _, r in df.iterrows()
    ]
    df["O"] = df.groupby("position")["O"].transform(lambda s: to_score_0_100(s))

    q = weights["quality"]
    df["Q"] = q["U"] * df["U"] + q["F"] * df["F"] + q["R"] * df["R"] + q["T"] * df["T"]

    # Value: xPts ~ points90 * expected minutes over horizon * O/50
    exp_mins = df["start_pct"].fillna(0) * 90 * settings.horizon_gws
    df["xpts"] = df["points90"].fillna(0) * (exp_mins / 90.0) * (df["O"] / 50.0).clip(0.3, 2.0)
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
    df["overall"] = ow["Q"] * df["Q"] + ow["M"] * df["M"] + ow["O"] * df["O"] + ow["V"] * df["V"]

    # Upside
    finishing = (df["npxg90"] - df["goals90"]).fillna(0)
    creation = (df["npxa90"] - df["assists90"]).fillna(0)
    min_trend = (df["minutes_pct_l5"] - df["minutes_pct"]).fillna(0)
    df["_fin"] = finishing
    df["_cre"] = creation
    df["_mtr"] = min_trend
    df = _norm_group(df, ["_fin", "_cre", "_mtr"])
    uwu = weights["upside"]
    df["upside"] = (
        uwu["finishing"] * df["_fin_s"]
        + uwu["creation"] * df["_cre_s"]
        + uwu["minutes_trend"] * df["_mtr_s"]
        + uwu["team_trend"] * 50.0
        + uwu["role_change"] * 50.0
        + uwu["price"] * 50.0
        + uwu["ownership"] * 50.0
    )

    def action(r) -> str:
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
            "xpts": round(float(r["xpts"]), 2),
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


def load_feats(conn: sqlite3.Connection, season: str, as_of_gw: int) -> pd.DataFrame:
    raw = pd.read_sql_query(
        "SELECT * FROM feat_player_asof WHERE season = ? AND as_of_gw = ?",
        conn,
        params=(season, as_of_gw),
    )
    if raw.empty:
        return raw
    extra = pd.json_normalize(raw["features_json"].map(json.loads))
    return pd.concat([raw.drop(columns=["features_json"]), extra], axis=1)


def score_asof(season: str, as_of_gw: int, settings: Settings | None = None) -> int:
    settings = settings or load_settings()
    weights = load_weights()
    conn = connect(settings)
    feats = load_feats(conn, season, as_of_gw)
    if feats.empty:
        conn.close()
        return 0
    scored = score_feature_frame(feats, weights, settings)
    persist_scores(conn, season, as_of_gw, scored)
    n = len(scored)
    conn.close()
    return n
