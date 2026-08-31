from __future__ import annotations

import argparse
import json
import sys

from fpl_model.config import load_settings
from fpl_model.db import connect, init_db
from fpl_model.ingest.vaastav import download_season, load_season
from fpl_model.features.asof import build_and_store
from fpl_model.scoring.engine import score_asof
from fpl_model.backtest.walkforward import walkforward


def _refresh_understat(season: str, settings, force: bool = True) -> None:
    from fpl_model.ingest.understat import build_and_store_map, download_league, store_understat_facts, write_id_map_review

    conn = connect(settings)
    init_db(conn, settings)
    league = download_league(season, settings, force=force)
    mapping = build_and_store_map(conn, season, league)
    matched = mapping["understat_id"].notna() & mapping["fpl_id"].notna()
    n_fpl = int(mapping["fpl_id"].notna().sum())
    n_matched = int(matched.sum())
    n_players, n_matches = store_understat_facts(conn, season, league, settings, force=force)
    print(
        f"understat map {n_matched}/{n_fpl} FPL players "
        f"({(n_matched / n_fpl * 100 if n_fpl else 0):.1f}%), "
        f"{n_players} player files, {n_matches} matches"
    )
    print(f"id map review -> {write_id_map_review(conn, season)}")
    conn.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fpl-model", description="FPL suitability model")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-db", help="Create SQLite schema")

    s = sub.add_parser("ingest", help="Load a season (vaastav archive or live FPL API)")
    s.add_argument("--season", required=True)
    s.add_argument("--source", choices=["auto", "vaastav", "live"], default="auto")
    s.add_argument("--download-only", action="store_true")
    s.add_argument("--skip-understat", action="store_true")
    s.add_argument("--force", action="store_true", help="Refetch live API / FBref caches")

    fb = sub.add_parser("fbref", help="Load FBref stats/defence/keepersadv/possession HTML and map to FPL ids")
    fb.add_argument("--season", required=True)
    fb.add_argument("--force", action="store_true")

    u = sub.add_parser("update", help="Refresh live FPL + Understat + FBref, then score the next 5 GWs")
    u.add_argument("--season", default=None)
    u.add_argument("--skip-understat", action="store_true")
    u.add_argument("--skip-fbref", action="store_true")

    f = sub.add_parser("features", help="Build as-of features for gameweek T (uses GW < T only)")
    f.add_argument("--season", required=True)
    f.add_argument("--as-of", type=int, required=True)

    sc = sub.add_parser("score", help="Score players at as-of GW T")
    sc.add_argument("--season", required=True)
    sc.add_argument("--as-of", type=int, required=True)

    c = sub.add_parser("card", help="Print a player card")
    c.add_argument("--season", required=True)
    c.add_argument("--as-of", type=int, required=True)
    c.add_argument("--name", required=True)

    ui = sub.add_parser("ui", help="Open the local player-card UI")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8765)

    xi = sub.add_parser("xi", help="Build a £100m squad + starting XI from as-of scores")
    xi.add_argument("--season", required=True)
    xi.add_argument("--as-of", type=int, required=True)
    xi.add_argument("--budget", type=float, default=100.0)
    xi.add_argument("--objective", choices=["xpts", "overall"], default="xpts")

    m = sub.add_parser("id-map", help="Rebuild FPL↔Understat map and write a review CSV")
    m.add_argument("--season", required=True)
    m.add_argument("--export-only", action="store_true", help="Write CSV from the current DB without rematching")

    e = sub.add_parser("errors", help="Top-20 overlap error analysis (overrank vs underrank)")
    e.add_argument("--season", required=True)
    e.add_argument("--as-of", default="10,20,30", help="Comma-separated as-of gameweeks")
    e.add_argument("--positions", default="MID,DEF")

    sb = sub.add_parser("squad-backtest", help="Walk-forward £100m xPts XI vs ownership template")
    sb.add_argument("--season", required=True)
    sb.add_argument("--min-gw", type=int, default=None)
    sb.add_argument("--max-gw", type=int, default=None)
    sb.add_argument("--horizon", type=int, default=None)
    sb.add_argument("--fast", action="store_true", help="Skip XI hill-climb (greedy fill only)")

    b = sub.add_parser("backtest", help="Walk-forward Spearman vs next-horizon actual points")
    b.add_argument("--season", required=True)
    b.add_argument("--min-gw", type=int, default=None)
    b.add_argument("--max-gw", type=int, default=None)
    b.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Prediction window in GWs (default: settings.horizon_gws). Non-default does not overwrite stored 5-GW scores.",
    )

    args = p.parse_args(argv)
    settings = load_settings()

    if args.cmd == "init-db":
        init_db(settings=settings)
        print(f"initialized {settings.db_path}")
        return 0

    if args.cmd == "ingest":
        source = args.source
        if source == "auto":
            source = "live" if args.season >= "2026-27" else "vaastav"
        if source == "live":
            from fpl_model.ingest.fpl_api import load_live_season

            season, as_of = load_live_season(args.season, settings, force=args.force)
            print(f"live FPL {season} loaded; next as-of GW{as_of}")
            if not args.skip_understat:
                _refresh_understat(season, settings)
            return 0
        download_season(args.season, settings)
        if args.download_only:
            print(f"downloaded {args.season} -> {settings.raw_dir / args.season}")
            return 0
        load_season(args.season, settings, skip_understat=args.skip_understat)
        print(f"loaded {args.season} into {settings.db_path}")
        return 0

    if args.cmd == "fbref":
        from fpl_model.ingest.fbref import load_fbref

        n = load_fbref(args.season, settings, force=args.force)
        print(f"fbref mapped {n} FPL players")
        return 0

    if args.cmd == "update":
        from fpl_model.ingest.fpl_api import load_live_season
        from fpl_model.ingest.fbref import load_fbref

        season, as_of = load_live_season(args.season, settings, force=True)
        print(f"live FPL {season}; scoring as-of GW{as_of} (next {settings.horizon_gws} GWs)")
        if not args.skip_understat:
            _refresh_understat(season, settings)
        if not args.skip_fbref:
            n_fb = load_fbref(season, settings, force=True)
            print(f"fbref mapped {n_fb} FPL players")
        n = build_and_store(season, as_of, settings)
        print(f"features as-of GW{as_of}: {n} players")
        n_s = score_asof(season, as_of, settings)
        print(f"scored {n_s} players")
        return 0

    if args.cmd == "features":
        n = build_and_store(args.season, args.as_of, settings)
        print(f"features as-of GW{args.as_of}: {n} players")
        return 0

    if args.cmd == "score":
        n = score_asof(args.season, args.as_of, settings)
        print(f"scored {n} players")
        return 0

    if args.cmd == "card":
        conn = connect(settings)
        q = """
        SELECT card_json, web_name, position, overall, action
        FROM score_player_asof
        WHERE season = ? AND as_of_gw = ? AND LOWER(web_name) LIKE ?
        ORDER BY overall DESC
        """
        rows = conn.execute(q, (args.season, args.as_of, f"%{args.name.lower()}%")).fetchall()
        conn.close()
        if not rows:
            print("no scores found; run `fpl-model score` first", file=sys.stderr)
            return 1
        for r in rows:
            card = json.loads(r["card_json"])
            print(f"{r['web_name']} ({r['position']})")
            print(f"  Player Quality {card['player_quality']}")
            print(f"    Underlying {card['underlying']}")
            print(f"    FPL Output  {card['fpl_output']}")
            print(f"    Role        {card['role']}")
            print(f"    Team        {card['team']}")
            print(f"  Minutes  {card['minutes']}")
            print(f"  Fixtures {card['fixtures']}")
            print(f"  Value    {card['value']}")
            print(f"  Overall  {card['overall']}")
            print(f"  Upside   {card['upside']}")
            if "upside_parts" in card:
                p = card["upside_parts"]
                print(
                    "    fin {finishing}  create {creation}  min {minutes_trend}  "
                    "team {team_trend}  role {role_change}  £ {price}  own {ownership}".format(**p)
                )
            print(f"  *{card['action']}*")
            print()
        return 0

    if args.cmd == "ui":
        from fpl_model.ui import serve

        serve(host=args.host, port=args.port, settings=settings)
        return 0

    if args.cmd == "xi":
        from fpl_model.ui.queries import build_xi

        conn = connect(settings)
        out = build_xi(conn, args.season, args.as_of, budget=args.budget, objective=args.objective)
        conn.close()
        if not out.get("ok"):
            print("could not build a full 15-man squad", file=sys.stderr)
            return 1
        print(
            f"{out['formation']} XI  spent £{out['spent']:.1f}  bank £{out['remaining']:.1f}  "
            f"obj={out['objective']} squad {out['squad_objective']}  xi {out['xi_objective']}"
        )
        if out.get("captain"):
            print(f"captain {out['captain']['web_name']}")
        print("\nXI:")
        for p in out["xi"]:
            print(f"  {p['position']:3} {p['web_name']:16} {p['team'] or '':3} £{p['price']:4.1f}  {out['objective']} {p[out['objective']]}")
        print("\nBench:")
        for p in out["bench"]:
            print(f"  {p['position']:3} {p['web_name']:16} {p['team'] or '':3} £{p['price']:4.1f}  {out['objective']} {p[out['objective']]}")
        return 0

    if args.cmd == "id-map":
        from fpl_model.ingest.understat import (
            build_and_store_map,
            download_league,
            write_id_map_review,
        )

        conn = connect(settings)
        init_db(conn, settings)
        if not args.export_only:
            league = download_league(args.season, settings)
            mapping = build_and_store_map(conn, args.season, league)
            matched = mapping["understat_id"].notna() & mapping["fpl_id"].notna()
            n_fpl = int(mapping["fpl_id"].notna().sum())
            print(f"mapped {int(matched.sum())}/{n_fpl} FPL players")
            print(mapping["match_method"].value_counts().to_string())
        dest = write_id_map_review(conn, args.season)
        conn.close()
        print(f"review CSV -> {dest}")
        return 0

    if args.cmd == "errors":
        from fpl_model.backtest.errors import run_error_analysis

        gws = [int(x.strip()) for x in args.as_of.split(",") if x.strip()]
        pos = [x.strip().upper() for x in args.positions.split(",") if x.strip()]
        reports = run_error_analysis(args.season, gws, pos, settings)
        for w in reports:
            if w.get("empty"):
                print(f"GW{w['as_of_gw']} {w['position']}: no scores (run backtest first)")
                continue
            print(f"\n=== GW{w['as_of_gw']} {w['position']} (n={w['n']}) ===")
            print(f"top-20 overlap  Overall {w['overlap_overall']:.0%}  |  xPts {w['overlap_xpts']:.0%}")
            print("Overranked (in our 20, not actual 20):")
            for r in w["overranked"][:6]:
                print(
                    f"  {r['web_name']:16} pred {int(r['pred_rank']):3}  actual {int(r['actual_rank']):3}  "
                    f"pts {r['actual_points']:4.0f}  M {r['M']:.0f}  R {r['R']:.0f}  "
                    f"defcon {r['defcon_hit_rate'] or 0:.0%}  {r['us_position']}"
                )
            print("Underranked (actual 20, not our 20):")
            for r in w["underranked"][:6]:
                print(
                    f"  {r['web_name']:16} pred {int(r['pred_rank']):3}  actual {int(r['actual_rank']):3}  "
                    f"pts {r['actual_points']:4.0f}  M {r['M']:.0f}  R {r['R']:.0f}  "
                    f"defcon {r['defcon_hit_rate'] or 0:.0%}  {r['us_position']}"
                )
        return 0

    if args.cmd == "squad-backtest":
        from fpl_model.backtest.squad import squad_walkforward, summarize_squad

        df = squad_walkforward(
            args.season,
            settings,
            args.min_gw,
            args.max_gw,
            horizon=args.horizon,
            polish=not args.fast,
        )
        if df.empty:
            print("no squad-backtest rows (ingest a season first)")
            return 1
        show = df.drop(columns=["xi", "need"], errors="ignore")
        print(show.to_string(index=False))
        print()
        s = summarize_squad(df)
        print(
            f"n={s['n']}  model {s.get('model_pts', 0):.1f}  template {s.get('template_pts', 0):.1f}  "
            f"delta {s.get('delta', 0):+.1f}  win {s.get('win_rate', 0):.0%}  "
            f"DEF £{s.get('def_spend') or 0:.1f}  FWD £{s.get('fwd_spend') or 0:.1f}"
        )
        return 0

    if args.cmd == "backtest":
        hz = args.horizon
        persist = hz is None
        df = walkforward(args.season, settings, args.min_gw, args.max_gw, horizon=hz, persist=persist)
        if df.empty:
            print("no backtest rows (ingest a season first)")
            return 1
        used = hz if hz is not None else settings.horizon_gws
        print(f"horizon={used} GW  persist={persist}")
        print(df.to_string(index=False))
        print()
        print(df.groupby("position")[["spearman", "top20_overlap", "xpts_mae"]].mean(numeric_only=True).to_string())
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
