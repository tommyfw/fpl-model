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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fpl-model", description="FPL suitability model")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-db", help="Create SQLite schema")

    s = sub.add_parser("ingest", help="Download vaastav + Understat and load a season")
    s.add_argument("--season", required=True)
    s.add_argument("--download-only", action="store_true")
    s.add_argument("--skip-understat", action="store_true")

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

    m = sub.add_parser("id-map", help="Rebuild FPL↔Understat map and write a review CSV")
    m.add_argument("--season", required=True)
    m.add_argument("--export-only", action="store_true", help="Write CSV from the current DB without rematching")

    b = sub.add_parser("backtest", help="Walk-forward Spearman vs next-horizon actual points")
    b.add_argument("--season", required=True)
    b.add_argument("--min-gw", type=int, default=None)
    b.add_argument("--max-gw", type=int, default=None)

    args = p.parse_args(argv)
    settings = load_settings()

    if args.cmd == "init-db":
        init_db(settings=settings)
        print(f"initialized {settings.db_path}")
        return 0

    if args.cmd == "ingest":
        download_season(args.season, settings)
        if args.download_only:
            print(f"downloaded {args.season} -> {settings.raw_dir / args.season}")
            return 0
        load_season(args.season, settings, skip_understat=args.skip_understat)
        print(f"loaded {args.season} into {settings.db_path}")
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
            print(f"  *{card['action']}*")
            print()
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

    if args.cmd == "backtest":
        df = walkforward(args.season, settings, args.min_gw, args.max_gw)
        if df.empty:
            print("no backtest rows (ingest a season first)")
            return 1
        print(df.to_string(index=False))
        print()
        print(df.groupby("position")[["spearman", "top20_overlap", "xpts_mae"]].mean(numeric_only=True).to_string())
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
