"""Read-only parent-weight grid. Never writes the live SQLite or config/weights.yaml."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from fpl_model.backtest.compare import _metrics
from fpl_model.backtest.walkforward import actual_points
from fpl_model.config import load_settings, load_weights
from fpl_model.features.asof import build_features_for_asof
from fpl_model.scoring.engine import load_feats, score_feature_frame

CACHE = HERE / "cache"
RESULTS = HERE / "results"


def _open_ro(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _load_grids() -> dict[str, Any]:
    return yaml.safe_load((HERE / "grids.yaml").read_text())


AS_IS = "yaml_as_is"

XPTS_CLIP = {
    "start_l5": (0.15, 0.85),
    "fixture_lo": (0.50, 1.00),
    "fixture_hi": (1.00, 1.60),
    "cs_scale": (0.25, 1.00),
    "bonus_scale": (0.15, 1.00),
    "bps_scale": (0.0, 0.40),
    "p90_residual": (0.0, 0.30),
    "p90_residual_def": (0.0, 0.20),
    "fwd_attack_scale": (0.85, 1.40),
    "defcon_def": (0.5, 2.5),
}


def _float_block(block: dict[str, Any] | None) -> dict[str, float]:
    return {str(k): float(v) for k, v in (block or {}).items() if v is not None}


def nudge_simplex(block: dict[str, Any], key: str, by: float) -> dict[str, float]:
    """Add `by` to `key`; take/give the mass from the other keys in proportion. Sum stays 1."""
    d = _float_block(block)
    if not d:
        return d
    total = sum(d.values())
    key = str(key)
    if key not in d:
        d[key] = 0.0
    d[key] = min(max(d[key] + float(by), 0.0), total)
    rest = total - d[key]
    others = [k for k in d if k != key]
    osum = sum(d[k] for k in others)
    if osum <= 1e-12:
        return {k: (total if k == key else 0.0) for k in d}
    for k in others:
        d[k] = rest * (d[k] / osum)
    return d


def apply_simplex_spec(block: dict[str, Any] | None, spec: dict[str, Any] | None) -> dict[str, float]:
    d = _float_block(block)
    if not spec:
        return d
    if spec.get("boost") is not None:
        return nudge_simplex(d, spec["boost"], float(spec.get("by") or 0))
    for k, v in spec.items():
        if k in ("boost", "by"):
            continue
        d[str(k)] = max(0.0, d.get(str(k), 0.0) + float(v))
    s = sum(d.values())
    if s > 0:
        d = {k: v / s for k, v in d.items()}
    return d


def apply_xpts_spec(block: dict[str, Any] | None, spec: dict[str, Any] | None) -> dict[str, Any]:
    d = dict(block or {})
    if not spec:
        return d
    for k, v in spec.items():
        if k in ("boost", "by"):
            continue
        cur = float(d[k]) if k in d and d[k] is not None else 0.0
        nxt = cur + float(v)
        bounds = XPTS_CLIP.get(str(k))
        if bounds:
            nxt = min(max(nxt, bounds[0]), bounds[1])
        d[k] = nxt
    return d


def apply_parent_mixes(
    base: dict[str, Any],
    overall: dict | None = None,
    quality: dict | None = None,
    xpts: dict | None = None,
) -> dict[str, Any]:
    w = deepcopy(base)
    if overall:
        w["overall"] = apply_simplex_spec(w.get("overall"), overall)
        by_pos = dict(w.get("overall_by_pos") or {})
        for pos, block in list(by_pos.items()):
            by_pos[pos] = apply_simplex_spec(block, overall)
        w["overall_by_pos"] = by_pos
    if quality:
        w["quality"] = apply_simplex_spec(w.get("quality"), quality)
        by_pos = dict(w.get("quality_by_pos") or {})
        for pos, block in list(by_pos.items()):
            by_pos[pos] = apply_simplex_spec(block, quality)
        w["quality_by_pos"] = by_pos
    if xpts:
        w["xpts"] = apply_xpts_spec(w.get("xpts"), xpts)
    return w


def parse_mix_label(label: str) -> tuple[str, str, str]:
    parts = str(label).split("__")
    while len(parts) < 3:
        parts.append(AS_IS)
    return parts[0], parts[1], parts[2]


def mix_relevant_for_pred(label: str, pred: str) -> bool:
    """Overall ranks only move with overall/quality; xPts only with the xpts overlay."""
    oname, qname, xname = parse_mix_label(label)
    if pred == "xpts":
        return oname == AS_IS and qname == AS_IS
    return xname == AS_IS


def ranking_settings(grids: dict[str, Any]) -> dict[str, Any]:
    r = grids.get("ranking") or {}
    raw_w = r.get("season_weights") or {"2024-25": 0.35, "2025-26": 0.65}
    return {
        "spearman_weight": float(r.get("spearman_weight", 0.7)),
        "top20_weight": float(r.get("top20_weight", 0.3)),
        "season_weights": {str(k): float(v) for k, v in raw_w.items()},
    }


def utilities_by_mix(rows: pd.DataFrame, pred: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """One row per mix: season utilities and a season-weighted final score (higher better).

    Per GW/position we already stored Spearman and Top-20. Average those over GWs,
    then u = spearman_w * Spearman + top20_w * Top-20, then mean over positions.
    Ranking on raw u keeps gaps between mixes; mean-of-ranks would flatten them.
    """
    if rows.empty or "pred" not in rows.columns:
        return pd.DataFrame()
    df = rows.loc[rows["pred"] == pred].copy()
    if df.empty:
        return df
    df = df.loc[df["mix"].map(lambda m: mix_relevant_for_pred(str(m), pred))]
    if df.empty:
        return df
    g = (
        df.groupby(["mix", "season", "position"], as_index=False)
        .agg(spearman=("spearman", "mean"), top20=("top20_overlap", "mean"))
    )
    sw = float(cfg["spearman_weight"])
    tw = float(cfg["top20_weight"])
    g["u"] = sw * g["spearman"] + tw * g["top20"]
    season_u = g.groupby(["mix", "season"], as_index=False)["u"].mean()
    wide = season_u.pivot(index="mix", columns="season", values="u")
    weights = cfg["season_weights"]
    finals = []
    for mix in wide.index:
        num = 0.0
        den = 0.0
        row: dict[str, Any] = {"mix": mix}
        for season, w in weights.items():
            val = wide.at[mix, season] if season in wide.columns else float("nan")
            key = "u_2425" if season == "2024-25" else ("u_2526" if season == "2025-26" else f"u_{season}")
            row[key] = None if pd.isna(val) else float(val)
            if pd.notna(val):
                num += w * float(val)
                den += w
        row["final_u"] = (num / den) if den else None
        finals.append(row)
    out = pd.DataFrame(finals)
    if out.empty or out["final_u"].isna().all():
        return out
    out["final_rank"] = out["final_u"].rank(ascending=False, method="min").astype("Int64")
    return out.sort_values(["final_rank", "mix"], na_position="last")


def _cache_path(season: str, gw: int) -> Path:
    return CACHE / f"{season}_gw{int(gw)}.pkl"


def load_or_build_feats(conn: sqlite3.Connection, season: str, gw: int, settings) -> pd.DataFrame:
    path = _cache_path(season, gw)
    if path.exists():
        return pd.read_pickle(path)
    feats = load_feats(conn, season, gw)
    if feats.empty:
        feats = build_features_for_asof(conn, season, gw, settings)
    if feats.empty:
        return feats
    CACHE.mkdir(parents=True, exist_ok=True)
    feats.to_pickle(path)
    return feats


def iter_candidates(base: dict[str, Any], grids: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    overalls = grids.get("overall") or {AS_IS: None}
    qualities = grids.get("quality") or {AS_IS: None}
    xptss = grids.get("xpts") or {AS_IS: None}
    out: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for oname, omix in overalls.items():
        for qname, qmix in qualities.items():
            label = f"{oname}__{qname}__{AS_IS}"
            if label in seen:
                continue
            seen.add(label)
            out.append((label, apply_parent_mixes(base, omix, qmix, None)))
    for xname, xmix in xptss.items():
        if xname == AS_IS:
            continue
        label = f"{AS_IS}__{AS_IS}__{xname}"
        if label in seen:
            continue
        seen.add(label)
        out.append((label, apply_parent_mixes(base, None, None, xmix)))
    return out


def run(db_path: Path | None = None) -> pd.DataFrame:
    settings = load_settings()
    db = Path(db_path) if db_path else settings.db_path
    if not db.exists():
        raise SystemExit(f"No database at {db}. Ingest 2024-25 / 2025-26 in the main project first.")
    grids = _load_grids()
    windows: dict[str, list[int]] = grids["windows"]
    base_w = load_weights()
    candidates = iter_candidates(base_w, grids)
    conn = _open_ro(db)
    rows = []
    try:
        for season, gws in windows.items():
            max_round = pd.read_sql_query(
                "SELECT MAX(round) AS m FROM fact_fpl_player_gw WHERE season = ?",
                conn,
                params=(season,),
            )["m"].iloc[0]
            if pd.isna(max_round):
                print(f"skip {season}: no GW facts in DB")
                continue
            end = int(max_round) - settings.horizon_gws + 1
            for t in gws:
                if t > end:
                    continue
                feats = load_or_build_feats(conn, season, t, settings)
                if feats.empty:
                    print(f"skip {season} GW{t}: no features")
                    continue
                actual = actual_points(conn, season, t, settings.horizon_gws)
                print(f"score {season} GW{t} n={len(feats)} mixes={len(candidates)}")
                for label, w in candidates:
                    scored = score_feature_frame(feats, w, settings)
                    for pred in ("overall", "xpts"):
                        met = _metrics(scored, actual, pred)
                        for _, r in met["by_pos"].iterrows():
                            rows.append(
                                {
                                    "mix": label,
                                    "season": season,
                                    "as_of_gw": t,
                                    "pred": pred,
                                    "position": r["position"],
                                    "spearman": r["spearman"],
                                    "top20_overlap": r["top20_overlap"],
                                }
                            )
    finally:
        conn.close()
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.groupby(["mix", "season", "pred"], as_index=False)
        .agg(spearman=("spearman", "mean"), top20=("top20_overlap", "mean"), n=("as_of_gw", "nunique"))
        .sort_values(["season", "pred", "spearman"], ascending=[True, True, False])
    )


def _fmt_u(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return "—"


def _dump_block(lines: list[str], title: str, mix_dict: dict | None) -> None:
    if mix_dict:
        lines.append(f"{title}:")
        for k, v in mix_dict.items():
            if isinstance(v, dict):
                lines.append(f"  {k}:")
                for kk, vv in v.items():
                    lines.append(f"    {kk}: {round(float(vv), 4) if isinstance(vv, (int, float)) else vv}")
            elif isinstance(v, float):
                lines.append(f"  {k}: {round(v, 4)}")
            else:
                lines.append(f"  {k}: {v}")
    else:
        lines.append(f"({title}: leave YAML as-is)")


def _xpts_diff(live: dict | None, chosen: dict | None) -> dict[str, Any] | None:
    live = live or {}
    chosen = chosen or {}
    out = {}
    for k, v in chosen.items():
        lv = live.get(k)
        try:
            if lv is None or abs(float(v) - float(lv)) > 1e-9:
                out[k] = v
        except (TypeError, ValueError):
            if v != lv:
                out[k] = v
    return out or None


def _resolved(base: dict[str, Any], grids: dict[str, Any], label: str) -> dict[str, Any] | None:
    return next((w for lab, w in iter_candidates(base, grids) if lab == label), None)


def _best_row(util: pd.DataFrame) -> pd.Series | None:
    if util.empty or util["final_u"].isna().all():
        return None
    return util.sort_values("final_u", ascending=False).iloc[0]


def holdout_report(rows: pd.DataFrame, grids: dict[str, Any]) -> str:
    cfg = ranking_settings(grids)
    sw = cfg["season_weights"]
    spw = cfg["spearman_weight"]
    tw = cfg["top20_weight"]
    lines = [
        f"Pick: {spw} Spearman + {tw} Top-20, mean over GK/DEF/MID/FWD, then",
        f"weighted seasons {sw} (higher score is better). Not mean-of-ranks.",
        "Overall and xPts are scored on separate mix sets (factored grid).",
        "",
    ]
    ov = utilities_by_mix(rows, "overall", cfg)
    xp = utilities_by_mix(rows, "xpts", cfg)
    best_ov = _best_row(ov)
    best_xp = _best_row(xp)
    if best_ov is None:
        lines.append("No Overall rows — ingest seasons and rerun.")
    else:
        lines.append(
            f"Best Overall: {best_ov['mix']}  final={best_ov['final_u']:.4f}"
            f"  24-25={_fmt_u(best_ov.get('u_2425'))}  25-26={_fmt_u(best_ov.get('u_2526'))}"
        )
        as_is = ov.loc[ov["mix"] == f"{AS_IS}__{AS_IS}__{AS_IS}"]
        if not as_is.empty:
            b = as_is.iloc[0]
            lines.append(
                f"YAML as-is Overall: final={b['final_u']:.4f}"
                f"  24-25={_fmt_u(b.get('u_2425'))}  25-26={_fmt_u(b.get('u_2526'))}"
            )
    if best_xp is None:
        lines.append("No xPts rows.")
    else:
        lines.append(
            f"Best xPts: {best_xp['mix']}  final={best_xp['final_u']:.4f}"
            f"  24-25={_fmt_u(best_xp.get('u_2425'))}  25-26={_fmt_u(best_xp.get('u_2526'))}"
        )
        as_is_x = xp.loc[xp["mix"] == f"{AS_IS}__{AS_IS}__{AS_IS}"]
        if not as_is_x.empty:
            b = as_is_x.iloc[0]
            lines.append(
                f"YAML as-is xPts: final={b['final_u']:.4f}"
                f"  24-25={_fmt_u(b.get('u_2425'))}  25-26={_fmt_u(b.get('u_2526'))}"
            )
    lines.append("")
    lines.append("Paste into config/weights.yaml only if you want this live (this lab will not do it):")
    base = load_weights()
    if best_ov is not None:
        oname, qname, _ = parse_mix_label(str(best_ov["mix"]))
        resolved = _resolved(base, grids, str(best_ov["mix"]))
        if oname == AS_IS and qname == AS_IS:
            _dump_block(lines, "overall", None)
            _dump_block(lines, "quality", None)
        elif resolved:
            _dump_block(lines, "overall", resolved.get("overall"))
            _dump_block(lines, "overall_by_pos", resolved.get("overall_by_pos"))
            if qname != AS_IS:
                _dump_block(lines, "quality", resolved.get("quality"))
                _dump_block(lines, "quality_by_pos", resolved.get("quality_by_pos"))
            else:
                _dump_block(lines, "quality", None)
    if best_xp is not None:
        _, _, xname = parse_mix_label(str(best_xp["mix"]))
        if xname == AS_IS:
            _dump_block(lines, "xpts overlay", None)
        else:
            resolved_x = _resolved(base, grids, str(best_xp["mix"]))
            _dump_block(lines, "xpts overlay", _xpts_diff(base.get("xpts"), (resolved_x or {}).get("xpts")))
    return "\n".join(lines) + "\n"


def suggested_yaml(rows: pd.DataFrame, grids: dict[str, Any]) -> str:
    cfg = ranking_settings(grids)
    ov = utilities_by_mix(rows, "overall", cfg)
    xp = utilities_by_mix(rows, "xpts", cfg)
    best_ov = _best_row(ov)
    best_xp = _best_row(xp)
    if best_ov is None and best_xp is None:
        return "# no grid rows\n"
    payload: dict[str, Any] = {
        "note": "Resolved onto live config/weights.yaml. Not applied to production. Copy by hand if you want it.",
        "ranking": cfg,
        "grid_specs": {
            "overall": {k: v for k, v in (grids.get("overall") or {}).items() if k != AS_IS},
            "quality": {k: v for k, v in (grids.get("quality") or {}).items() if k != AS_IS},
            "xpts": {k: v for k, v in (grids.get("xpts") or {}).items() if k != AS_IS},
        },
    }
    base = load_weights()
    if best_ov is not None:
        oname, qname, _ = parse_mix_label(str(best_ov["mix"]))
        payload["overall_mix"] = str(best_ov["mix"])
        payload["overall_final_u"] = float(best_ov["final_u"])
        resolved = _resolved(base, grids, str(best_ov["mix"]))
        if resolved and not (oname == AS_IS and qname == AS_IS):
            payload["overall"] = resolved.get("overall")
            payload["overall_by_pos"] = resolved.get("overall_by_pos")
            if qname != AS_IS:
                payload["quality"] = resolved.get("quality")
                payload["quality_by_pos"] = resolved.get("quality_by_pos")
    if best_xp is not None:
        _, _, xname = parse_mix_label(str(best_xp["mix"]))
        payload["xpts_mix"] = str(best_xp["mix"])
        payload["xpts_final_u"] = float(best_xp["final_u"])
        if xname != AS_IS:
            resolved_x = _resolved(base, grids, str(best_xp["mix"]))
            payload["xpts"] = _xpts_diff(base.get("xpts"), (resolved_x or {}).get("xpts"))
    return yaml.safe_dump(payload, sort_keys=False)


def main() -> int:
    p = argparse.ArgumentParser(description="Isolated parent-weight grid (read-only DB).")
    p.add_argument("--db", type=Path, default=None, help="Main fpl.sqlite (default: config/settings.yaml)")
    p.add_argument("--cache-only", action="store_true", help="Build pickle caches, do not score mixes")
    args = p.parse_args()
    settings = load_settings()
    db = args.db or settings.db_path
    if args.cache_only:
        grids = _load_grids()
        conn = _open_ro(db)
        try:
            for season, gws in grids["windows"].items():
                for t in gws:
                    n = len(load_or_build_feats(conn, season, t, settings))
                    print(f"cache {season} GW{t}: {n} players")
        finally:
            conn.close()
        return 0
    df = run(db)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "grid_rows.csv", index=False)
    summary = summarize(df)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    grids = _load_grids()
    (RESULTS / "holdout.txt").write_text(holdout_report(df, grids), encoding="utf-8")
    (RESULTS / "suggested.yaml").write_text(suggested_yaml(df, grids), encoding="utf-8")
    print(summary.to_string(index=False))
    print()
    print((RESULTS / "holdout.txt").read_text())
    print(f"Wrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
