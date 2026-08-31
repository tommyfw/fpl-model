from __future__ import annotations

import json
import sqlite3

import pandas as pd

from fpl_model.config import Settings, load_settings
from fpl_model.db import connect
from fpl_model.backtest.walkforward import actual_points


def _feat_fields(card_or_feat: str | None, keys: list[str]) -> dict:
    if not card_or_feat:
        return {}
    try:
        obj = json.loads(card_or_feat)
    except json.JSONDecodeError:
        return {}
    return {k: obj.get(k) for k in keys}


def slice_table(conn: sqlite3.Connection, season: str, as_of_gw: int, position: str, horizon: int, k: int = 20) -> pd.DataFrame:
    scores = pd.read_sql_query(
        """
        SELECT s.fpl_id, s.web_name, s.position, s.overall, s.U, s.F, s.R, s.M, s.O, s.V, s.xpts, f.features_json
        FROM score_player_asof s
        LEFT JOIN feat_player_asof f
          ON f.season = s.season AND f.as_of_gw = s.as_of_gw AND f.fpl_id = s.fpl_id
        WHERE s.season = ? AND s.as_of_gw = ? AND s.position = ?
        """,
        conn,
        params=(season, as_of_gw, position),
    )
    if scores.empty:
        return scores
    actual = actual_points(conn, season, as_of_gw, horizon)
    df = scores.merge(actual, on="fpl_id", how="left")
    df["actual_points"] = df["actual_points"].fillna(0)
    extra = df["features_json"].map(
        lambda x: _feat_fields(x, ["defcon_hit_rate", "us_position", "start_pct", "minutes", "pen_score", "pitch_attack"])
    )
    extra_df = pd.DataFrame(list(extra), index=df.index)
    df = pd.concat([df.drop(columns=["features_json"]), extra_df], axis=1)
    df["pred_rank"] = df["overall"].rank(ascending=False, method="min")
    df["xpts_rank"] = df["xpts"].rank(ascending=False, method="min")
    df["actual_rank"] = df["actual_points"].rank(ascending=False, method="min")
    return df


def overlap(ids_a, ids_b, k: int = 20) -> float:
    return len(set(ids_a) & set(ids_b)) / float(k)


def analyse_window(conn: sqlite3.Connection, season: str, as_of_gw: int, position: str, settings: Settings) -> dict:
    k = 20
    df = slice_table(conn, season, as_of_gw, position, settings.horizon_gws, k)
    if df.empty:
        return {"as_of_gw": as_of_gw, "position": position, "empty": True}
    by_overall = df.nsmallest(k, "pred_rank")
    by_xpts = df.nsmallest(k, "xpts_rank")
    by_actual = df.nsmallest(k, "actual_rank")
    actual_ids = set(by_actual["fpl_id"])
    over = by_overall.loc[~by_overall["fpl_id"].isin(actual_ids)].copy()
    under = by_actual.loc[~by_actual["fpl_id"].isin(set(by_overall["fpl_id"]))].copy()
    over = over.sort_values("pred_rank")
    under = under.sort_values("actual_rank")

    def pack(frame: pd.DataFrame, n: int = 8) -> list[dict]:
        cols = [
            "web_name",
            "pred_rank",
            "actual_rank",
            "actual_points",
            "overall",
            "xpts",
            "U",
            "R",
            "M",
            "defcon_hit_rate",
            "us_position",
            "start_pct",
        ]
        out = []
        for _, r in frame.head(n).iterrows():
            row = {}
            for c in cols:
                v = r.get(c)
                if pd.isna(v):
                    row[c] = None
                elif hasattr(v, "item"):
                    row[c] = v.item()
                else:
                    row[c] = v
            out.append(row)
        return out

    return {
        "as_of_gw": as_of_gw,
        "position": position,
        "n": int(len(df)),
        "overlap_overall": overlap(by_overall["fpl_id"], actual_ids, k),
        "overlap_xpts": overlap(by_xpts["fpl_id"], actual_ids, k),
        "overranked": pack(over),
        "underranked": pack(under),
        "over_mean": {
            "defcon_hit_rate": float(over["defcon_hit_rate"].mean()) if "defcon_hit_rate" in over and over["defcon_hit_rate"].notna().any() else None,
            "start_pct": float(over["start_pct"].mean()) if over["start_pct"].notna().any() else None,
            "M": float(over["M"].mean()) if len(over) else None,
            "R": float(over["R"].mean()) if len(over) else None,
            "actual_points": float(over["actual_points"].mean()) if len(over) else None,
        },
        "under_mean": {
            "defcon_hit_rate": float(under["defcon_hit_rate"].mean()) if "defcon_hit_rate" in under and under["defcon_hit_rate"].notna().any() else None,
            "start_pct": float(under["start_pct"].mean()) if under["start_pct"].notna().any() else None,
            "M": float(under["M"].mean()) if len(under) else None,
            "R": float(under["R"].mean()) if len(under) else None,
            "actual_points": float(under["actual_points"].mean()) if len(under) else None,
        },
    }


def run_error_analysis(
    season: str,
    as_of: list[int],
    positions: list[str],
    settings: Settings | None = None,
) -> list[dict]:
    settings = settings or load_settings()
    conn = connect(settings)
    out = [analyse_window(conn, season, t, pos, settings) for t in as_of for pos in positions]
    conn.close()
    return out
