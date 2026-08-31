from __future__ import annotations

import sqlite3
from dataclasses import replace

import pandas as pd
from scipy.stats import spearmanr

from fpl_model.config import Settings, load_settings, load_weights
from fpl_model.db import connect
from fpl_model.features.asof import build_features_for_asof, persist_features
from fpl_model.scoring.engine import persist_scores, score_feature_frame


def actual_points(conn: sqlite3.Connection, season: str, start_gw: int, horizon: int) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT fpl_id, SUM(total_points) AS actual_points, SUM(minutes) AS actual_minutes
        FROM fact_fpl_player_gw
        WHERE season = ? AND round >= ? AND round < ?
        GROUP BY fpl_id
        """,
        conn,
        params=(season, start_gw, start_gw + horizon),
    )


def walkforward(
    season: str,
    settings: Settings | None = None,
    min_gw: int | None = None,
    max_gw: int | None = None,
    horizon: int | None = None,
    persist: bool | None = None,
) -> pd.DataFrame:
    settings = settings or load_settings()
    if horizon is not None:
        settings = replace(settings, horizon_gws=horizon)
    if persist is None:
        persist = horizon is None
    weights = load_weights()
    conn = connect(settings)
    max_round = pd.read_sql_query(
        "SELECT MAX(round) AS m FROM fact_fpl_player_gw WHERE season = ?",
        conn,
        params=(season,),
    )["m"].iloc[0]
    if pd.isna(max_round):
        conn.close()
        return pd.DataFrame()
    start = min_gw or settings.min_as_of_gw
    end = min(max_gw or settings.max_as_of_gw, int(max_round) - settings.horizon_gws + 1)
    rows = []
    for t in range(start, end + 1):
        feats = build_features_for_asof(conn, season, t, settings)
        if feats.empty:
            continue
        scored = score_feature_frame(feats, weights, settings)
        if persist:
            persist_features(conn, season, t, feats)
            persist_scores(conn, season, t, scored)
        actual = actual_points(conn, season, t, settings.horizon_gws)
        merged = scored.merge(actual, on="fpl_id", how="left")
        merged["actual_points"] = merged["actual_points"].fillna(0)
        for pos, g in merged.groupby("position"):
            if g["overall"].nunique() < 3 or g["actual_points"].nunique() < 3:
                rho = None
            else:
                rho, _ = spearmanr(g["overall"], g["actual_points"])
            top = g.nlargest(20, "overall")
            actual_top = set(g.nlargest(20, "actual_points")["fpl_id"])
            overlap = len(set(top["fpl_id"]) & actual_top) / 20.0
            mae = (g["xpts"] - g["actual_points"]).abs().mean()
            rows.append(
                {
                    "season": season,
                    "as_of_gw": t,
                    "position": pos,
                    "n": len(g),
                    "spearman": None if rho is None or pd.isna(rho) else float(rho),
                    "top20_overlap": overlap,
                    "xpts_mae": float(mae),
                }
            )
    conn.close()
    return pd.DataFrame(rows)
