from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from fpl_model.ingest.id_map import team_key


def previous_season(season: str) -> str:
    y = int(season.split("-")[0])
    return f"{y - 1}-{str(y)[-2:]}"


def promoted_strength_anchor(by_key: pd.DataFrame, quantile: float = 0.25) -> tuple[float, float]:
    """Bottom-of-table PL rates for clubs with no prior PL season.

    ``prior_attack`` is xG for; ``prior_defence`` is −xGA. Quantile 0.25 is the
    weaker attacking and weaker defending quarter of last year's league.
    """
    q = float(np.clip(quantile, 0.0, 1.0))
    if by_key.empty:
        return 1.05, -1.55
    att = float(by_key["prior_attack"].quantile(q))
    de = float(by_key["prior_defence"].quantile(q))
    return att, de


def load_team_priors(
    conn: sqlite3.Connection, season: str, promoted_quantile: float = 0.25
) -> pd.DataFrame:
    """Prior team attack/defence (xG for / −xGA per match) from the previous PL season.

    Returning clubs use their own prior-season Understat rates.
    Promoted clubs (no prior PL season) use the previous-season PL quantile
    (default: bottom quartile), not the league mean.
    Index: team_id for ``season``.
    """
    prev = previous_season(season)
    curr = pd.read_sql_query(
        "SELECT team_id, name FROM dim_team WHERE season = ?",
        conn,
        params=(season,),
    )
    if curr.empty:
        return pd.DataFrame()
    curr = curr.copy()
    curr["tkey"] = curr["name"].map(team_key)

    ut = pd.read_sql_query(
        """
        SELECT team_key, is_home, xg_for, xg_against
        FROM fact_understat_team_match
        WHERE season = ? AND finished = 1
          AND xg_for IS NOT NULL AND xg_against IS NOT NULL
        """,
        conn,
        params=(prev,),
    )
    venue_cols = [
        "prior_attack_home",
        "prior_attack_away",
        "prior_defence_home",
        "prior_defence_away",
    ]
    if ut.empty:
        # Fallback: scoreline rates from prior FPL fixtures
        fx = pd.read_sql_query(
            """
            SELECT team_h, team_a, team_h_score, team_a_score
            FROM fact_fixture
            WHERE season = ? AND finished = 1
              AND team_h_score IS NOT NULL AND team_a_score IS NOT NULL
            """,
            conn,
            params=(prev,),
        )
        prev_teams = pd.read_sql_query(
            "SELECT team_id, name FROM dim_team WHERE season = ?",
            conn,
            params=(prev,),
        )
        if fx.empty or prev_teams.empty:
            return pd.DataFrame()
        prev_teams = prev_teams.copy()
        prev_teams["tkey"] = prev_teams["name"].map(team_key)
        rows = []
        for _, r in fx.iterrows():
            rows.append(
                {
                    "team_id": r["team_h"],
                    "is_home": 1,
                    "gf": r["team_h_score"],
                    "ga": r["team_a_score"],
                }
            )
            rows.append(
                {
                    "team_id": r["team_a"],
                    "is_home": 0,
                    "gf": r["team_a_score"],
                    "ga": r["team_h_score"],
                }
            )
        raw = pd.DataFrame(rows)
        g = raw.groupby("team_id", as_index=False).agg(n=("gf", "count"), gf=("gf", "sum"), ga=("ga", "sum"))
        g["prior_attack"] = g["gf"] / g["n"].clip(lower=1)
        g["prior_defence"] = -(g["ga"] / g["n"].clip(lower=1))
        g = g.merge(prev_teams[["team_id", "tkey"]], on="team_id")
        by_key = g.set_index("tkey")[["prior_attack", "prior_defence"]]
        vg = raw.groupby(["team_id", "is_home"], as_index=False).agg(
            n=("gf", "count"), gf=("gf", "sum"), ga=("ga", "sum")
        )
        vg["prior_attack"] = vg["gf"] / vg["n"].clip(lower=1)
        vg["prior_defence"] = -(vg["ga"] / vg["n"].clip(lower=1))
        vg = vg.merge(prev_teams[["team_id", "tkey"]], on="team_id")
        by_key = _attach_venue_priors(by_key, vg.set_index(["tkey", "is_home"]))
    else:
        ut = ut.copy()
        ut["is_home"] = pd.to_numeric(ut["is_home"], errors="coerce").fillna(0).astype(int)
        g = ut.groupby("team_key", as_index=True).agg(
            n=("xg_for", "count"), xgf=("xg_for", "sum"), xga=("xg_against", "sum")
        )
        g["n"] = g["n"].clip(lower=1)
        by_key = pd.DataFrame(
            {
                "prior_attack": g["xgf"] / g["n"],
                "prior_defence": -(g["xga"] / g["n"]),
            },
            index=g.index,
        )
        vg = ut.groupby(["team_key", "is_home"], as_index=True).agg(
            n=("xg_for", "count"), xgf=("xg_for", "sum"), xga=("xg_against", "sum")
        )
        vg["n"] = vg["n"].clip(lower=1)
        vg["prior_attack"] = vg["xgf"] / vg["n"]
        vg["prior_defence"] = -(vg["xga"] / vg["n"])
        by_key = _attach_venue_priors(by_key, vg)

    promo_att, promo_de = promoted_strength_anchor(by_key, promoted_quantile)
    promo_venue = {
        col: float(by_key[col].quantile(promoted_quantile))
        if col in by_key.columns and by_key[col].notna().any()
        else (promo_att if "attack" in col else promo_de)
        for col in venue_cols
    }

    tkey = curr.set_index("team_id")["tkey"]
    out = curr.set_index("team_id")[[]]
    out["prior_attack"] = tkey.map(by_key["prior_attack"])
    out["prior_defence"] = tkey.map(by_key["prior_defence"])
    for col in venue_cols:
        if col in by_key.columns:
            out[col] = tkey.map(by_key[col])
        else:
            out[col] = np.nan
    out["is_promoted"] = out["prior_attack"].isna().astype(int)
    out["prior_attack"] = out["prior_attack"].fillna(promo_att)
    out["prior_defence"] = out["prior_defence"].fillna(promo_de)
    for col in venue_cols:
        out[col] = out[col].fillna(promo_venue[col])
        overall = out["prior_attack"] if "attack" in col else out["prior_defence"]
        out[col] = out[col].fillna(overall)
    return out


def _attach_venue_priors(by_key: pd.DataFrame, vg: pd.DataFrame) -> pd.DataFrame:
    """Add last-season home/away xG columns onto a team-key prior table."""
    out = by_key.copy()
    if vg.empty:
        return out
    idx = vg.index
    if not isinstance(idx, pd.MultiIndex) or idx.nlevels < 2:
        return out

    def _col(is_home: int, src: str) -> pd.Series:
        try:
            return vg.xs(int(is_home), level=-1)[src]
        except (KeyError, TypeError, ValueError):
            return pd.Series(dtype=float)

    out["prior_attack_home"] = _col(1, "prior_attack")
    out["prior_attack_away"] = _col(0, "prior_attack")
    out["prior_defence_home"] = _col(1, "prior_defence")
    out["prior_defence_away"] = _col(0, "prior_defence")
    return out


def load_player_priors(conn: sqlite3.Connection, season: str) -> pd.DataFrame:
    """Per-90 rates from the previous season, keyed by this season's fpl_id via FPL `code`."""
    prev = previous_season(season)
    codes = pd.read_sql_query(
        "SELECT fpl_id, code FROM dim_player WHERE season = ? AND code IS NOT NULL",
        conn,
        params=(season,),
    )
    if codes.empty:
        return pd.DataFrame()
    prev_players = pd.read_sql_query(
        "SELECT fpl_id, code FROM dim_player WHERE season = ? AND code IS NOT NULL",
        conn,
        params=(prev,),
    )
    if prev_players.empty:
        return pd.DataFrame()
    gw = pd.read_sql_query(
        """
        SELECT fpl_id, SUM(minutes) AS minutes, SUM(total_points) AS points,
               SUM(goals_scored) AS goals, SUM(assists) AS assists,
               SUM(bonus) AS bonus, SUM(bps) AS bps,
               SUM(expected_goals) AS xg, SUM(expected_assists) AS xa,
               SUM(expected_goal_involvements) AS xgi,
               SUM(expected_goals_conceded) AS xgc, SUM(saves) AS saves,
               SUM(threat) AS threat, SUM(creativity) AS creativity,
               SUM(starts) AS starts, COUNT(DISTINCT round) AS matches
        FROM fact_fpl_player_gw
        WHERE season = ?
        GROUP BY fpl_id
        """,
        conn,
        params=(prev,),
    )
    if gw.empty:
        return pd.DataFrame()
    gw = gw.merge(prev_players, on="fpl_id")
    rates = pd.DataFrame({"code": gw["code"], "minutes": gw["minutes"]})
    for col, src in [
        ("points90", "points"),
        ("goals90", "goals"),
        ("assists90", "assists"),
        ("bonus90", "bonus"),
        ("bps90", "bps"),
        ("npxg90", "xg"),
        ("npxa90", "xa"),
        ("npxgi90", "xgi"),
        ("xgc90", "xgc"),
        ("saves90", "saves"),
        ("threat90", "threat"),
        ("creativity90", "creativity"),
    ]:
        rates[col] = np.where(gw["minutes"] > 0, gw[src] / gw["minutes"] * 90.0, 0.0)
    rates["start_pct"] = gw["starts"] / gw["matches"].clip(lower=1)
    rates["minutes_pct"] = gw["minutes"] / (90.0 * gw["matches"].clip(lower=1))
    link = codes.merge(rates, on="code", how="inner")
    return link.drop(columns=["code"]).drop_duplicates("fpl_id").set_index("fpl_id")
