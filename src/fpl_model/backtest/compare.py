from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pandas as pd
from scipy.stats import spearmanr

from fpl_model.config import Settings, load_settings, load_weights
from fpl_model.db import connect
from fpl_model.features.asof import build_features_for_asof
from fpl_model.scoring.engine import score_feature_frame


def _metrics(scored: pd.DataFrame, actual: pd.DataFrame, pred_col: str) -> dict[str, float]:
    merged = scored.merge(actual, on="fpl_id", how="left")
    merged["actual_points"] = merged["actual_points"].fillna(0)
    rows = []
    for pos, g in merged.groupby("position"):
        if g[pred_col].nunique() < 3 or g["actual_points"].nunique() < 3:
            rho = None
        else:
            rho, _ = spearmanr(g[pred_col], g["actual_points"])
        top = set(g.nlargest(20, pred_col)["fpl_id"])
        actual_top = set(g.nlargest(20, "actual_points")["fpl_id"])
        rows.append(
            {
                "position": pos,
                "spearman": float(rho) if rho is not None and pd.notna(rho) else None,
                "top20_overlap": len(top & actual_top) / 20.0,
            }
        )
    out = pd.DataFrame(rows)
    return {
        "spearman_mean": float(out["spearman"].mean(skipna=True)),
        "top20_mean": float(out["top20_overlap"].mean()),
        "by_pos": out,
    }


def run_backtest(
    season: str,
    settings: Settings | None = None,
    weights: dict[str, Any] | None = None,
    *,
    min_gw: int | None = None,
    max_gw: int | None = None,
    horizon: int | None = None,
) -> pd.DataFrame:
    """Walk-forward metrics for Overall and xPts (does not persist)."""
    settings = settings or load_settings()
    if horizon is not None:
        settings = replace(settings, horizon_gws=horizon)
    weights = weights or load_weights()
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
    from fpl_model.backtest.walkforward import actual_points

    rows = []
    for t in range(start, end + 1):
        feats = build_features_for_asof(conn, season, t, settings)
        if feats.empty:
            continue
        scored = score_feature_frame(feats, weights, settings)
        actual = actual_points(conn, season, t, settings.horizon_gws)
        for pred in ("overall", "xpts"):
            m = _metrics(scored, actual, pred)
            for _, r in m["by_pos"].iterrows():
                rows.append(
                    {
                        "season": season,
                        "as_of_gw": t,
                        "pred": pred,
                        "position": r["position"],
                        "spearman": r["spearman"],
                        "top20_overlap": r["top20_overlap"],
                    }
                )
    conn.close()
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.groupby(["season", "pred", "position"], as_index=False)[["spearman", "top20_overlap"]]
        .mean()
        .sort_values(["season", "pred", "position"])
    )


def mean_score(summary: pd.DataFrame, season: str, pred: str = "overall") -> float:
    sub = summary[(summary["season"] == season) & (summary["pred"] == pred)]
    if sub.empty:
        return float("nan")
    return float(sub["spearman"].mean())


def with_overall_weights(base: dict[str, Any], q: float, m: float, o: float, v: float) -> dict[str, Any]:
    w = deepcopy(base)
    w["overall"] = {"Q": q, "M": m, "O": o, "V": v}
    return w
