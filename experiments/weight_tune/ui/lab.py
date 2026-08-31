"""In-memory experimental scores. Never writes live SQLite or production weights."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from run import (
    CACHE,
    RESULTS,
    apply_parent_mixes,
    iter_candidates,
    load_or_build_feats,
    mix_relevant_for_pred,
    ranking_settings,
    utilities_by_mix,
    _load_grids,
    _open_ro,
)

from fpl_model.config import load_settings, load_weights
from fpl_model.scoring.engine import score_feature_frame
from fpl_model.squad.optimize import build_squad

_SCORED: dict[tuple[str, int, str], pd.DataFrame] = {}
_TEAMS: dict[str, dict[int, dict[str, str]]] = {}


def mix_labels() -> list[str]:
    return [label for label, _ in iter_candidates(load_weights(), _load_grids())]


def snapshot_list() -> list[dict[str, Any]]:
    out = []
    if CACHE.exists():
        for p in sorted(CACHE.glob("*_gw*.pkl")):
            stem = p.stem
            if "_gw" not in stem:
                continue
            season, gw = stem.rsplit("_gw", 1)
            try:
                out.append({"season": season, "as_of_gw": int(gw), "n": 0, "source": "cache"})
            except ValueError:
                continue
    if not out:
        grids = _load_grids()
        for season, gws in (grids.get("windows") or {}).items():
            for t in gws:
                out.append({"season": season, "as_of_gw": int(t), "n": 0, "source": "grid"})
    return out


def _teams(season: str) -> dict[int, dict[str, str]]:
    if season in _TEAMS:
        return _TEAMS[season]
    settings = load_settings()
    m: dict[int, dict[str, str]] = {}
    if settings.db_path.exists():
        conn = _open_ro(settings.db_path)
        try:
            for r in conn.execute("SELECT team_id, name, short_name FROM dim_team WHERE season = ?", (season,)):
                m[int(r["team_id"])] = {"name": r["name"] or "", "short_name": r["short_name"] or ""}
        finally:
            conn.close()
    _TEAMS[season] = m
    return m


def scored_frame(season: str, as_of: int, mix: str) -> pd.DataFrame:
    key = (season, int(as_of), mix)
    if key in _SCORED:
        return _SCORED[key]
    settings = load_settings()
    conn = _open_ro(settings.db_path) if settings.db_path.exists() else sqlite3.connect(":memory:")
    try:
        feats = load_or_build_feats(conn, season, as_of, settings)
    finally:
        conn.close()
    if feats.empty:
        return feats
    weights = load_weights()
    chosen = next((w for label, w in iter_candidates(weights, _load_grids()) if label == mix), None)
    if chosen is None:
        chosen = apply_parent_mixes(weights, None, None)
    scored = score_feature_frame(feats, chosen, settings)
    teams = _teams(season)
    if "team_id" in scored.columns:
        scored["team"] = scored["team_id"].map(lambda i: teams.get(int(i), {}).get("short_name") if pd.notna(i) else "")
        scored["team_name"] = scored["team_id"].map(lambda i: teams.get(int(i), {}).get("name") if pd.notna(i) else "")
    _SCORED[key] = scored
    return scored


def list_players(
    season: str,
    as_of: int,
    mix: str,
    *,
    position: str | None = None,
    q: str | None = None,
    sort: str = "overall",
    limit: int = 400,
) -> list[dict[str, Any]]:
    df = scored_frame(season, as_of, mix)
    if df.empty:
        return []
    work = df.copy()
    if position:
        work = work.loc[work["position"] == position.upper()]
    if q:
        name = work["web_name"].astype(str).str.lower() if "web_name" in work.columns else pd.Series("", index=work.index)
        work = work.loc[name.str.contains(q.lower(), na=False)]
    col = sort if sort in work.columns else "overall"
    if sort == "quality":
        col = "Q"
    if sort == "value":
        col = "V"
    if sort == "minutes":
        col = "M"
    if sort == "fixtures":
        col = "O"
    work = work.sort_values(col, ascending=False, na_position="last").head(int(limit))
    return [_row_series(pd.Series(rec)) for rec in work.to_dict(orient="records")]


def get_player(season: str, as_of: int, mix: str, fpl_id: int) -> dict[str, Any] | None:
    df = scored_frame(season, as_of, mix)
    if df.empty or "fpl_id" not in df.columns:
        return None
    hit = df.loc[df["fpl_id"] == int(fpl_id)]
    if hit.empty:
        return None
    r = hit.iloc[0]
    out = _row_series(r)
    out["card"] = {
        "player_quality": _f(r.get("Q")),
        "underlying": _f(r.get("U")),
        "fpl_output": _f(r.get("F")),
        "role": _f(r.get("R")),
        "team": _f(r.get("T")),
        "minutes": _f(r.get("M")),
        "fixtures": _f(r.get("O")),
        "value": _f(r.get("V")),
        "overall": _f(r.get("overall")),
        "upside": _f(r.get("upside")),
        "xpts": round(_f(r.get("xpts"), 0) or 0, 2),
        "action": r.get("action"),
        "upside_parts": {
            "finishing": _f(r.get("up_finishing"), 50),
            "creation": _f(r.get("up_creation"), 50),
            "minutes_trend": _f(r.get("up_minutes_trend"), 50),
            "team_trend": _f(r.get("up_team_trend"), 50),
            "role_change": _f(r.get("up_role_change"), 50),
            "price": _f(r.get("up_price"), 50),
            "ownership": _f(r.get("up_ownership"), 50),
        },
    }
    feats = r.to_dict() if hasattr(r, "to_dict") else {}
    out["stats"] = {
        "minutes": _f(r.get("minutes")),
        "starts": _f(r.get("starts")),
        "points": _f(r.get("points")),
        "goals": _f(r.get("goals")),
        "assists": _f(r.get("assists")),
        "bonus": _f(r.get("bonus")),
        "npxg": _f(r.get("npxg")),
        "npxa": _f(r.get("npxa")),
        "npxg90": _f(r.get("npxg90")),
        "npxa90": _f(r.get("npxa90")),
        "defcon_hit_rate": _f(r.get("defcon_hit_rate")),
        "xg_source": r.get("xg_source") or "fpl",
    }
    return out


def xi(season: str, as_of: int, mix: str, budget: float = 100.0, objective: str = "xpts") -> dict[str, Any]:
    players = list_players(season, as_of, mix, sort=objective, limit=2000)
    result = build_squad(players, budget=budget, objective=objective)

    def slim(p: dict[str, Any]) -> dict[str, Any]:
        return {
            "fpl_id": p["fpl_id"],
            "web_name": p.get("web_name"),
            "position": p.get("position"),
            "team": p.get("team"),
            "price": p.get("price"),
            "xpts": p.get("xpts"),
            "overall": p.get("overall"),
            "action": p.get("action"),
        }

    return {
        "season": season,
        "as_of_gw": as_of,
        "mix": mix,
        "ok": result.get("ok"),
        "budget": result.get("budget"),
        "spent": result.get("spent"),
        "remaining": result.get("remaining"),
        "objective": result.get("objective"),
        "formation": result.get("formation"),
        "squad_objective": result.get("squad_objective"),
        "xi_objective": result.get("xi_objective"),
        "captain": slim(result["captain"]) if result.get("captain") else None,
        "xi": [slim(p) for p in result.get("xi") or []],
        "bench": [slim(p) for p in result.get("bench") or []],
        "error": result.get("error"),
    }


def _rank_mixes_by_position(pred: str) -> list[dict[str, Any]]:
    path = RESULTS / "grid_rows.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty or "pred" not in df.columns:
        return []
    df = df.loc[df["pred"] == pred]
    df = df.loc[df["mix"].map(lambda m: mix_relevant_for_pred(str(m), pred))]
    if df.empty:
        return []
    g = (
        df.groupby(["mix", "season", "position"], as_index=False)
        .agg(spearman=("spearman", "mean"), top20=("top20_overlap", "mean"))
    )
    seasons = ("2024-25", "2025-26")
    keys = {"2024-25": "s2425", "2025-26": "s2526"}
    boards = []
    for pos in ("GK", "DEF", "MID", "FWD"):
        sub = g.loc[g["position"] == pos]
        if sub.empty:
            continue
        ranked: dict[str, pd.DataFrame] = {}
        for season in seasons:
            s = sub.loc[sub["season"] == season].copy()
            if s.empty:
                continue
            n = max(int(s["mix"].nunique()), 1)
            s["spearman_rank"] = s["spearman"].rank(ascending=False, method="min").astype(int)
            s["top20_rank"] = s["top20"].rank(ascending=False, method="min").astype(int)
            s["n_mixes"] = n
            ranked[season] = s.set_index("mix")
        mixes = sorted(sub["mix"].astype(str).unique())
        n_mixes = len(mixes)
        rows = []
        for mix in mixes:
            row: dict[str, Any] = {"mix": mix, "n_mixes": n_mixes}
            for season, prefix in keys.items():
                if season not in ranked or mix not in ranked[season].index:
                    continue
                r = ranked[season].loc[mix]
                row[f"{prefix}_spearman"] = float(r["spearman"])
                row[f"{prefix}_spearman_rank"] = int(r["spearman_rank"])
                row[f"{prefix}_top20"] = float(r["top20"])
                row[f"{prefix}_top20_rank"] = int(r["top20_rank"])
            rows.append(row)
        rows.sort(
            key=lambda x: (
                x.get("s2526_spearman_rank") or 999,
                x.get("s2425_spearman_rank") or 999,
                x["mix"],
            )
        )
        boards.append({"position": pos, "n_mixes": n_mixes, "rows": rows})
    return boards


def _season_table(rows: pd.DataFrame, pred: str, u_key: str) -> list[dict[str, Any]]:
    """Rank mixes on that season's raw utility (same formula as Final, one season only)."""
    cfg = ranking_settings(_load_grids())
    util = utilities_by_mix(rows, pred, cfg)
    if util.empty or u_key not in util.columns:
        return []
    work = util.loc[util[u_key].notna()].copy()
    if work.empty:
        return []
    n = int(work["mix"].nunique())
    work["season_rank"] = work[u_key].rank(ascending=False, method="min").astype(int)
    out = []
    for r in work.to_dict(orient="records"):
        out.append(
            {
                "mix": r["mix"],
                "n_mixes": n,
                "season_rank": int(r["season_rank"]),
                "score": r[u_key],
            }
        )
    out.sort(key=lambda x: (x["season_rank"], x["mix"]))
    return out


def _final_table(rows: pd.DataFrame, pred: str) -> list[dict[str, Any]]:
    cfg = ranking_settings(_load_grids())
    util = utilities_by_mix(rows, pred, cfg)
    if util.empty:
        return []
    n = int(util["mix"].nunique())
    out = []
    for r in util.to_dict(orient="records"):
        out.append(
            {
                "mix": r["mix"],
                "n_mixes": n,
                "final_rank": int(r["final_rank"]) if pd.notna(r.get("final_rank")) else None,
                "final_u": r.get("final_u"),
                "u_2425": r.get("u_2425"),
                "u_2526": r.get("u_2526"),
            }
        )
    out.sort(key=lambda x: (x["final_rank"] or 999, x["mix"]))
    return out


def metrics() -> dict[str, Any]:
    summary_path = RESULTS / "summary.csv"
    rows_path = RESULTS / "grid_rows.csv"
    holdout = RESULTS / "holdout.txt"
    suggested = RESULTS / "suggested.yaml"
    summary = []
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        summary = df.to_dict(orient="records")
    rows = pd.read_csv(rows_path) if rows_path.exists() else pd.DataFrame()
    ov = _rank_mixes_by_position("overall")
    xp = _rank_mixes_by_position("xpts")
    cfg = ranking_settings(_load_grids())
    return {
        "summary": summary,
        "holdout": holdout.read_text(encoding="utf-8") if holdout.exists() else "Run python run.py first.",
        "suggested": yaml.safe_load(suggested.read_text()) if suggested.exists() else None,
        "ranking": cfg,
        "ranks_overall": ov,
        "ranks_xpts": xp,
        "combined_overall_2425": _season_table(rows, "overall", "u_2425"),
        "combined_overall_2526": _season_table(rows, "overall", "u_2526"),
        "combined_xpts_2425": _season_table(rows, "xpts", "u_2425"),
        "combined_xpts_2526": _season_table(rows, "xpts", "u_2526"),
        "final_overall": _final_table(rows, "overall"),
        "final_xpts": _final_table(rows, "xpts"),
    }


def _f(v: Any, default: float | None = None) -> float | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _row_series(r: Any) -> dict[str, Any]:
    get = r.get if hasattr(r, "get") else lambda k, d=None: getattr(r, k, d)
    tid = get("team_id")
    try:
        tid = int(tid) if pd.notna(tid) else None
    except (TypeError, ValueError):
        tid = None
    return {
        "fpl_id": int(get("fpl_id")),
        "web_name": get("web_name"),
        "position": get("position"),
        "team_id": tid,
        "team": get("team"),
        "team_name": get("team_name"),
        "price": _f(get("price")),
        "overall": _f(get("overall")),
        "upside": _f(get("upside")),
        "xpts": _f(get("xpts")),
        "action": get("action"),
        "status": get("status") or "a",
        "Q": _f(get("Q")),
        "U": _f(get("U")),
        "F": _f(get("F")),
        "R": _f(get("R")),
        "T": _f(get("T")),
        "M": _f(get("M")),
        "O": _f(get("O")),
        "V": _f(get("V")),
    }
