from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd
from scipy.stats import spearmanr

from fpl_model.config import Settings, load_settings, load_weights
from fpl_model.db import connect
from fpl_model.features.asof import build_features_for_asof, persist_features
from fpl_model.ingest.priors import load_team_priors
from fpl_model.scoring.engine import persist_scores, score_feature_frame


def actual_points(conn: sqlite3.Connection, season: str, start_gw: int, horizon: int) -> pd.DataFrame:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(fact_fpl_player_gw)").fetchall()}
    xp_sel = "SUM(xp) AS official_xp" if "xp" in cols else "NULL AS official_xp"
    return pd.read_sql_query(
        f"""
        SELECT fpl_id, SUM(total_points) AS actual_points, SUM(minutes) AS actual_minutes,
               {xp_sel}
        FROM fact_fpl_player_gw
        WHERE season = ? AND round >= ? AND round < ?
        GROUP BY fpl_id
        """,
        conn,
        params=(season, start_gw, start_gw + horizon),
    )


def calibrate_asof(
    season: str,
    as_of_gw: int,
    settings: Settings | None = None,
    *,
    persist: bool = True,
) -> dict:
    """Stored `xpts_next` at as-of T vs actual FPL points in GW T (first week of the horizon)."""
    settings = settings or load_settings()
    conn = connect(settings)
    scores = pd.read_sql_query(
        """
        SELECT fpl_id, position, team_id, xpts, card_json
        FROM score_player_asof
        WHERE season = ? AND as_of_gw = ?
        """,
        conn,
        params=(season, as_of_gw),
    )
    actual = pd.read_sql_query(
        """
        SELECT fpl_id, total_points AS actual_next, minutes, was_home, opponent_team
        FROM fact_fpl_player_gw
        WHERE season = ? AND round = ?
        """,
        conn,
        params=(season, as_of_gw),
    )
    if scores.empty:
        conn.close()
        return {"empty": True, "reason": "no stored scores"}
    if actual.empty:
        conn.close()
        return {"empty": True, "reason": "no actual GW points"}
    actual["actual_next"] = pd.to_numeric(actual["actual_next"], errors="coerce")
    if actual["actual_next"].notna().sum() == 0:
        conn.close()
        return {
            "empty": True,
            "reason": "actual total_points not in the DB for this GW (live ingest may still be empty)",
        }

    def _next(raw: str) -> float:
        try:
            card = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return float("nan")
        v = card.get("xpts_next", card.get("xpts"))
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    def _unmatched(raw: str) -> bool:
        try:
            card = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return False
        flags = card.get("flags") or {}
        return flags.get("understat") is False

    scores["pred"] = scores["card_json"].map(_next)
    scores["unmatched"] = scores["card_json"].map(_unmatched)
    if "was_home" in actual.columns:
        actual = actual.sort_values("fpl_id")
        first = actual.groupby("fpl_id", as_index=False).agg(
            actual_next=("actual_next", "sum"),
            minutes=("minutes", "sum"),
            was_home=("was_home", "max"),
            opponent_team=("opponent_team", "first"),
        )
    else:
        first = actual.groupby("fpl_id", as_index=False).agg(
            actual_next=("actual_next", "sum"),
            minutes=("minutes", "sum"),
        )
    merged = scores.merge(first, on="fpl_id", how="inner")
    merged = merged.dropna(subset=["pred", "actual_next"])
    if merged.empty:
        conn.close()
        return {"empty": True, "reason": "no overlap with actuals"}

    def _slice(g: pd.DataFrame, label: str) -> dict:
        err = g["pred"] - g["actual_next"]
        return {
            "label": label,
            "n": int(len(g)),
            "mae": float(err.abs().mean()),
            "bias": float(err.mean()),
        }

    by_pos = []
    for pos, g in merged.groupby("position"):
        row = _slice(g, str(pos))
        row["position"] = pos
        row["unmatched"] = int(g["unmatched"].sum())
        by_pos.append(row)

    by_venue = []
    if "was_home" in merged.columns:
        home_m = pd.to_numeric(merged["was_home"], errors="coerce").fillna(0) >= 1
        if home_m.any():
            by_venue.append(_slice(merged.loc[home_m], "home"))
        if (~home_m).any():
            by_venue.append(_slice(merged.loc[~home_m], "away"))

    by_opp = []
    priors = load_team_priors(conn, season)
    if not priors.empty and "is_promoted" in priors.columns and "opponent_team" in merged.columns:
        promo = set(
            int(i)
            for i, v in priors["is_promoted"].items()
            if pd.notna(v) and float(v) >= 1
        )
        opp = pd.to_numeric(merged["opponent_team"], errors="coerce")
        vs_p = opp.isin(promo)
        if vs_p.any():
            by_opp.append(_slice(merged.loc[vs_p], "vs_promoted"))
        if (~vs_p).any():
            by_opp.append(_slice(merged.loc[~vs_p], "vs_established"))

    table = calibration_table(merged["pred"], merged["actual_next"])
    pack = {
        "empty": False,
        "season": season,
        "as_of_gw": as_of_gw,
        "n": len(merged),
        "mae": float((merged["pred"] - merged["actual_next"]).abs().mean()),
        "bias": float((merged["pred"] - merged["actual_next"]).mean()),
        "unmatched": int(merged["unmatched"].sum()),
        "by_position": pd.DataFrame(by_pos),
        "by_venue": by_venue,
        "by_opponent": by_opp,
        "table": table,
    }
    if persist:
        persist_calibration(conn, pack)
    conn.close()
    return pack


def persist_calibration(conn: sqlite3.Connection, pack: dict) -> None:
    if pack.get("empty"):
        return
    breakdown = {
        "by_position": pack["by_position"].to_dict(orient="records")
        if isinstance(pack.get("by_position"), pd.DataFrame)
        else pack.get("by_position") or [],
        "by_venue": pack.get("by_venue") or [],
        "by_opponent": pack.get("by_opponent") or [],
        "table": pack["table"].to_dict(orient="records")
        if isinstance(pack.get("table"), pd.DataFrame) and not pack["table"].empty
        else [],
    }
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_calibrate_asof (
            season TEXT NOT NULL,
            as_of_gw INTEGER NOT NULL,
            n INTEGER,
            mae REAL,
            bias REAL,
            unmatched INTEGER,
            breakdown_json TEXT,
            computed_at TEXT,
            PRIMARY KEY (season, as_of_gw)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO fact_calibrate_asof(season, as_of_gw, n, mae, bias, unmatched, breakdown_json, computed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(season, as_of_gw) DO UPDATE SET
            n = excluded.n,
            mae = excluded.mae,
            bias = excluded.bias,
            unmatched = excluded.unmatched,
            breakdown_json = excluded.breakdown_json,
            computed_at = excluded.computed_at
        """,
        (
            pack["season"],
            int(pack["as_of_gw"]),
            int(pack["n"]),
            float(pack["mae"]),
            float(pack["bias"]),
            int(pack["unmatched"]),
            json.dumps(breakdown),
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
    )
    conn.commit()


def calibration_table(pred: pd.Series, actual: pd.Series, n_bins: int = 5) -> pd.DataFrame:
    """Mean predicted vs actual by predicted-xPts quantile (reliability)."""
    d = pd.DataFrame({"pred": pd.to_numeric(pred, errors="coerce"), "actual": pd.to_numeric(actual, errors="coerce")})
    d = d.dropna()
    if len(d) < n_bins * 8:
        return pd.DataFrame()
    d["bin"] = pd.qcut(d["pred"], n_bins, duplicates="drop")
    return (
        d.groupby("bin", observed=True)
        .agg(n=("pred", "size"), pred_mean=("pred", "mean"), actual_mean=("actual", "mean"))
        .reset_index()
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
            bias = float((g["xpts"] - g["actual_points"]).mean())
            xp_rho = None
            if "official_xp" in g.columns and g["official_xp"].notna().sum() > 8 and g["official_xp"].nunique() > 3:
                xp_rho, _ = spearmanr(g["official_xp"], g["actual_points"])
                if xp_rho is not None and pd.isna(xp_rho):
                    xp_rho = None
            rows.append(
                {
                    "season": season,
                    "as_of_gw": t,
                    "position": pos,
                    "n": len(g),
                    "spearman": None if rho is None or pd.isna(rho) else float(rho),
                    "top20_overlap": overlap,
                    "xpts_mae": float(mae),
                    "xpts_bias": bias,
                    "xp_spearman": None if xp_rho is None else float(xp_rho),
                }
            )
    conn.close()
    return pd.DataFrame(rows)
