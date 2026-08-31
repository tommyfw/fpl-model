"""Compare prior/minutes settings and overall-weight grids without persisting scores."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pandas as pd

from fpl_model.backtest.compare import _metrics
from fpl_model.backtest.walkforward import actual_points
from fpl_model.config import load_settings, load_weights
from fpl_model.db import connect
from fpl_model.features.asof import build_features_for_asof
from fpl_model.scoring.engine import score_feature_frame

# Sparse GWs: early season (prior-sensitive) + mid/late. Fast enough to iterate.
WINDOWS = {
    "2025-26": [2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32],
    "2024-25": [2, 4, 8, 12, 16, 20, 24, 28, 32],
}

OVERALL_GRIDS = [
    ("ow_current", {"Q": 0.48, "M": 0.27, "O": 0.18, "V": 0.07}),
    ("ow_more_m", {"Q": 0.42, "M": 0.34, "O": 0.17, "V": 0.07}),
    ("ow_more_q", {"Q": 0.55, "M": 0.23, "O": 0.16, "V": 0.06}),
    ("ow_more_o", {"Q": 0.45, "M": 0.25, "O": 0.22, "V": 0.08}),
]


def _settings_variants(base):
    return {
        "legacy_prior": replace(
            base,
            max_prior_weight=1.0,
            max_playing_prior_weight=1.0,
            prior_playing_grace_matches=99,
            dampen_unused_minutes=False,
        ),
        "caps_only": replace(base, dampen_unused_minutes=False),
        "caps_and_dampen": base,
    }


def run_tune() -> pd.DataFrame:
    base = load_settings()
    weights = load_weights()
    conn = connect(base)
    rows = []
    try:
        for sname, settings in _settings_variants(base).items():
            for season, gws in WINDOWS.items():
                max_round = pd.read_sql_query(
                    "SELECT MAX(round) AS m FROM fact_fpl_player_gw WHERE season = ?",
                    conn,
                    params=(season,),
                )["m"].iloc[0]
                if pd.isna(max_round):
                    continue
                end = int(max_round) - settings.horizon_gws + 1
                for t in gws:
                    if t > end:
                        continue
                    feats = build_features_for_asof(conn, season, t, settings)
                    if feats.empty:
                        continue
                    actual = actual_points(conn, season, t, settings.horizon_gws)
                    for wname, ow in OVERALL_GRIDS:
                        w = deepcopy(weights)
                        w["overall"] = ow
                        scored = score_feature_frame(feats, w, settings)
                        for pred in ("overall", "xpts"):
                            met = _metrics(scored, actual, pred)
                            early = t < 8
                            rows.append(
                                {
                                    "settings": sname,
                                    "weights": wname,
                                    "season": season,
                                    "as_of_gw": t,
                                    "early": early,
                                    "pred": pred,
                                    "spearman_mean": met["spearman_mean"],
                                    "top20_mean": met["top20_mean"],
                                }
                            )
                            for _, r in met["by_pos"].iterrows():
                                rows.append(
                                    {
                                        "settings": sname,
                                        "weights": wname,
                                        "season": season,
                                        "as_of_gw": t,
                                        "early": early,
                                        "pred": pred,
                                        "spearman_mean": r["spearman"],
                                        "top20_mean": r["top20_overlap"],
                                        "position": r["position"],
                                    }
                                )
    finally:
        conn.close()
    return pd.DataFrame(rows)


def _print_block(title: str, df: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    if df.empty:
        print("(empty)")
        return
    g = (
        df[df["position"].isna()]
        .groupby(["settings", "weights", "season"], as_index=False)[["spearman_mean", "top20_mean"]]
        .mean()
        .sort_values(["season", "spearman_mean"], ascending=[True, False])
    )
    print(g.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main() -> None:
    df = run_tune()
    if df.empty:
        print("no tune rows (ingest seasons first)")
        return
    core = df[df["pred"] == "overall"]
    _print_block("Overall Spearman (all sampled GWs)", core)
    _print_block("Overall Spearman (GW < 8)", core[core["early"]])
    _print_block("Overall Spearman (GW >= 8)", core[~core["early"]])
    x = df[df["pred"] == "xpts"]
    _print_block("xPts Spearman (all sampled GWs)", x)

    # Pick config that does not hurt either season vs legacy, prefer higher mean.
    both = (
        core[core["position"].isna()]
        .groupby(["settings", "weights", "season"], as_index=False)["spearman_mean"]
        .mean()
    )
    wide = both.pivot_table(index=["settings", "weights"], columns="season", values="spearman_mean")
    print("\n=== Combined (must have both seasons) ===")
    print(wide.to_string(float_format=lambda v: f"{v:.3f}"))
    if {"2024-25", "2025-26"}.issubset(wide.columns):
        wide["min_season"] = wide[["2024-25", "2025-26"]].min(axis=1)
        wide["mean_season"] = wide[["2024-25", "2025-26"]].mean(axis=1)
        print("\nBest by min(season) then mean:")
        print(wide.sort_values(["min_season", "mean_season"], ascending=False).head(8).to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
