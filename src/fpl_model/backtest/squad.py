from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd

from fpl_model.config import Settings, load_settings, load_weights
from fpl_model.db import connect
from fpl_model.features.asof import build_features_for_asof
from fpl_model.scoring.engine import score_feature_frame
from fpl_model.squad.optimize import SQUAD_NEED, build_squad, pick_xi


def _rows(scored: pd.DataFrame) -> list[dict[str, Any]]:
    cols = [
        "fpl_id",
        "web_name",
        "position",
        "team_id",
        "price",
        "status",
        "xpts",
        "overall",
        "last_selected",
    ]
    keep = [c for c in cols if c in scored.columns]
    return scored[keep].to_dict("records")


def _template_pool(players: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ownership as the optimiser objective → highest-owned legal 15."""
    out = []
    for p in players:
        q = dict(p)
        own = p.get("last_selected")
        try:
            val = float(own)
            q["xpts"] = 0.0 if val != val else val
        except (TypeError, ValueError):
            q["xpts"] = 0.0
        out.append(q)
    return out


def xi_actual_points(
    xi: list[dict[str, Any]],
    captain: dict[str, Any] | None,
    actual: dict[int, float],
) -> float:
    total = 0.0
    cap_id = captain["fpl_id"] if captain else None
    for p in xi:
        fid = int(p["fpl_id"])
        pts = float(actual.get(fid, 0.0))
        total += pts
        if cap_id is not None and fid == int(cap_id):
            total += pts
    return total


def _pos_spend(squad: list[dict[str, Any]], pos: str) -> float:
    return sum(float(p.get("price") or 0) for p in squad if p.get("position") == pos)


def evaluate_window(
    scored: pd.DataFrame,
    actual_map: dict[int, float],
    *,
    budget: float = 100.0,
    polish: bool = True,
) -> dict[str, Any]:
    players = _rows(scored)
    model = build_squad(players, budget=budget, objective="xpts", polish=polish)
    template = build_squad(_template_pool(players), budget=budget, objective="xpts", polish=False)
    # Template XI by ownership (already encoded as xpts); captain = most owned starter.
    if template.get("ok") and template.get("xi"):
        t_pack = pick_xi(template["squad"], objective="xpts")
        template["xi"] = t_pack["xi"]
        template["captain"] = t_pack["captain"]
        template["formation"] = t_pack["formation"]

    model_pts = xi_actual_points(model.get("xi") or [], model.get("captain"), actual_map) if model.get("ok") else None
    templ_pts = xi_actual_points(template.get("xi") or [], template.get("captain"), actual_map) if template.get("ok") else None
    xi_names = [p.get("web_name") for p in (model.get("xi") or [])]
    return {
        "model_ok": bool(model.get("ok")),
        "template_ok": bool(template.get("ok")),
        "model_pts": model_pts,
        "template_pts": templ_pts,
        "delta": None if model_pts is None or templ_pts is None else model_pts - templ_pts,
        "model_formation": model.get("formation"),
        "template_formation": template.get("formation"),
        "model_spent": model.get("spent"),
        "def_spend": _pos_spend(model.get("squad") or [], "DEF") if model.get("ok") else None,
        "fwd_spend": _pos_spend(model.get("squad") or [], "FWD") if model.get("ok") else None,
        "xi": xi_names,
        "need": dict(SQUAD_NEED),
    }


def squad_walkforward(
    season: str,
    settings: Settings | None = None,
    min_gw: int | None = None,
    max_gw: int | None = None,
    horizon: int | None = None,
    *,
    polish: bool = True,
) -> pd.DataFrame:
    """£100m xPts XI vs highest-owned template, scored on actual points (captain 2×)."""
    settings = settings or load_settings()
    if horizon is not None:
        settings = replace(settings, horizon_gws=horizon)
    weights = load_weights()
    conn = connect(settings)
    from fpl_model.backtest.walkforward import actual_points

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
        actual = actual_points(conn, season, t, settings.horizon_gws)
        amap = {
            int(fid): float(pts)
            for fid, pts in zip(actual["fpl_id"], actual["actual_points"], strict=True)
        }
        pack = evaluate_window(scored, amap, polish=polish)
        rows.append({"season": season, "as_of_gw": t, **pack})
    conn.close()
    return pd.DataFrame(rows)


def summarize_squad(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"n": 0}
    sub = df.dropna(subset=["delta"])
    if sub.empty:
        return {"n": 0}
    return {
        "n": int(len(sub)),
        "model_pts": float(sub["model_pts"].mean()),
        "template_pts": float(sub["template_pts"].mean()),
        "delta": float(sub["delta"].mean()),
        "win_rate": float((sub["delta"] > 0).mean()),
        "def_spend": float(sub["def_spend"].mean()) if "def_spend" in sub else None,
        "fwd_spend": float(sub["fwd_spend"].mean()) if "fwd_spend" in sub else None,
    }
