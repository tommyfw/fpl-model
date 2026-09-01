from __future__ import annotations

import re
import sqlite3
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_model.config import Settings, load_settings
from fpl_model.db import connect, init_db
from fpl_model.ingest.id_map import last_name, normalize_name, team_key
from fpl_model.ingest.vaastav import _to_sql

FBREF = "https://fbref.com/en/comps/9"


def fbref_season_path(season: str) -> str:
    y0 = season.split("-")[0]
    y1 = season.split("-")[1]
    y1 = y1 if len(y1) == 4 else f"20{y1}"
    return f"{y0}-{y1}"


def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(x) for x in t if str(x) != "nan").strip() for t in df.columns]
    df.columns = [re.sub(r"^Unnamed: \d+_level_\d+_", "", str(c)) for c in df.columns]
    return df


def _tables_from_html(html: str) -> list[pd.DataFrame]:
    chunks = [html] + re.findall(r"<!--(.*?)-->", html, flags=re.S)
    for table_id in (
        "stats_standard",
        "stats_defense",
        "stats_keeper_adv",
        "stats_possession",
        "stats_squads_possession_for",
    ):
        m = re.search(rf'<table[^>]*id=["\']{table_id}["\'][\s\S]*?</table>', html, flags=re.I)
        if m:
            chunks.append(m.group(0))
    tables: list[pd.DataFrame] = []
    for chunk in chunks:
        try:
            tables.extend(pd.read_html(StringIO(chunk), flavor="lxml"))
        except ValueError:
            continue
    return [_flatten_cols(t) for t in tables]


def _pick_table(tables: list[pd.DataFrame], need: set[str]) -> pd.DataFrame | None:
    best = None
    best_key = (-1, -1)
    for t in tables:
        cols = {str(c) for c in t.columns}
        hit = sum(any(n.lower() in str(c).lower() for c in cols) for n in need)
        if hit <= 0:
            continue
        density = int(t.notna().to_numpy().sum())
        key = (hit, density)
        if key > best_key:
            best, best_key = t, key
    return best


def _col(df: pd.DataFrame, *names: str) -> pd.Series:
    lower = {str(c).lower(): c for c in df.columns}
    for n in names:
        for k, orig in lower.items():
            if n.lower() == k or k.endswith("_" + n.lower()) or n.lower() in k:
                return pd.to_numeric(df[orig], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _looks_like_fbref(html: str) -> bool:
    if not html or "Just a moment" in html:
        return False
    return "stats_standard" in html or "stats_defense" in html or "stats_keeper_adv" in html or "stats_possession" in html or 'id="stats_' in html


def _fetch_html(url: str) -> str:
    """Try Chromium first (Cloudflare). Empty string if the challenge does not clear."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return ""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page(
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                    locale="en-GB",
                )
                page.goto(url, wait_until="commit", timeout=20_000)
                page.wait_for_timeout(4_000)
                html = page.content()
                return html if _looks_like_fbref(html) else ""
            finally:
                browser.close()
    except Exception:
        return ""


def download_fbref(season: str, settings: Settings | None = None, force: bool = False) -> Path:
    settings = settings or load_settings()
    dest = settings.data_dir / "raw" / "fbref" / season
    dest.mkdir(parents=True, exist_ok=True)
    slug = fbref_season_path(season)
    pages = {
        "stats": [
            f"{FBREF}/stats/Premier-League-Stats",
            f"{FBREF}/{slug}/stats/{slug}-Premier-League-Stats",
        ],
        "defense": [
            f"{FBREF}/defense/Premier-League-Stats",
            f"{FBREF}/{slug}/defense/{slug}-Premier-League-Stats",
        ],
        "keepersadv": [
            f"{FBREF}/keepersadv/Premier-League-Stats",
            f"{FBREF}/{slug}/keepersadv/{slug}-Premier-League-Stats",
        ],
        "possession": [
            f"{FBREF}/possession/Premier-League-Stats",
            f"{FBREF}/{slug}/possession/{slug}-Premier-League-Stats",
        ],
    }
    blocked = False
    for i, (name, urls) in enumerate(pages.items()):
        path = dest / f"{name}.html"
        if path.exists() and path.stat().st_size > 20_000 and _looks_like_fbref(path.read_text(encoding="utf-8", errors="ignore")) and not force:
            continue
        html = ""
        for url in urls:
            html = _fetch_html(url)
            if html:
                path.write_text(html, encoding="utf-8")
                break
        if not html:
            blocked = True
        if i + 1 < len(pages):
            time.sleep(3.5)
    if blocked:
        print(
            "FBref is behind Cloudflare from this environment. "
            "In a normal browser, save these pages as HTML into:\n"
            f"  {dest}/stats.html\n"
            f"  {dest}/defense.html\n"
            f"  {dest}/keepersadv.html\n"
            f"  {dest}/possession.html\n"
            f"  {pages['stats'][0]}\n"
            f"  {pages['defense'][0]}\n"
            f"  {pages['keepersadv'][0]}\n"
            f"  {pages['possession'][0]}\n"
            "Then rerun: fpl-model fbref --season " + season
        )
    return dest


def _player_col(df: pd.DataFrame) -> str:
    return next(c for c in df.columns if "player" in str(c).lower())


def _squad_col(df: pd.DataFrame) -> str | None:
    return next((c for c in df.columns if "squad" in str(c).lower()), None)


def _stat_rows(html: str, table_id: str) -> list[dict[str, str]]:
    m = re.search(rf'<table[^>]*id=["\']{table_id}["\'][\s\S]*?</table>', html, flags=re.I)
    if not m:
        return []
    rows: list[dict[str, str]] = []
    for tr in re.findall(r"<tr[^>]*>([\s\S]*?)</tr>", m.group(0), flags=re.I):
        cells: dict[str, str] = {}
        for stat, inner in re.findall(r'data-stat="([^"]+)"[^>]*>([\s\S]*?)</t[dh]>', tr, flags=re.I):
            cells[stat] = re.sub(r"<[^>]+>", "", inner).replace("\xa0", " ").strip()
        if cells:
            rows.append(cells)
    return rows


def _num(rows: list[dict[str, str]], *keys: str) -> pd.Series:
    def one(r: dict[str, str]) -> float:
        for k in keys:
            v = r.get(k, "")
            if v in ("", "NA", "Matches"):
                continue
            try:
                return float(v.replace(",", ""))
            except ValueError:
                continue
        return float("nan")

    return pd.Series([one(r) for r in rows], dtype=float)


def parse_squad_possession(dest: Path) -> pd.DataFrame:
    path = dest / "possession.html"
    if not path.exists():
        return pd.DataFrame()
    html = path.read_text(encoding="utf-8")
    rows = _stat_rows(html, "stats_squads_possession_for")
    if rows and any(r.get("team") and r.get("team") != "Squad" for r in rows):
        squad = pd.Series([r.get("team") or r.get("squad") or "" for r in rows])
        att = _num(rows, "touches_att_3rd")
        de = _num(rows, "touches_def_3rd")
        att_pen = _num(rows, "touches_att_pen_area")
        poss = _num(rows, "possession")
        denom = (att + de).replace(0, np.nan)
        out = pd.DataFrame(
            {
                "squad": squad.astype(str),
                "poss": poss,
                "att3rd": att,
                "def3rd": de,
                "att_pen": att_pen,
                "tilt": att / denom,
            }
        )
        out = out.loc[~out["squad"].str.contains("Squad", na=False) & (out["squad"] != "")]
        if not out.empty:
            out["tilt"] = out["tilt"].where(out["tilt"].notna(), out["poss"] / 100.0)
            return out.drop_duplicates("squad")
    tables = _tables_from_html(html)
    squads = []
    for t in tables:
        cols = {str(c).lower() for c in t.columns}
        if any("player" in c for c in cols):
            continue
        if not any("squad" in c or c == "squad" for c in cols):
            continue
        if not any("att 3rd" in c or "poss" in c for c in cols):
            continue
        squads.append(t)
    if not squads:
        return pd.DataFrame()
    sq = max(squads, key=lambda t: int(t.notna().to_numpy().sum())).copy()
    sc = _squad_col(sq) or next(c for c in sq.columns if "squad" in str(c).lower() or str(c).lower() == "squad")
    att = _col(sq, "Touches_Att 3rd", "Att 3rd")
    de = _col(sq, "Touches_Def 3rd", "Def 3rd")
    att_pen = _col(sq, "Touches_Att Pen", "Att Pen")
    poss = _col(sq, "Poss")
    denom = (att + de).replace(0, np.nan)
    out = pd.DataFrame(
        {
            "squad": sq[sc].astype(str),
            "poss": poss,
            "att3rd": att,
            "def3rd": de,
            "att_pen": att_pen,
            "tilt": att / denom,
        }
    )
    out = out.loc[~out["squad"].str.contains("Squad", na=False)].drop_duplicates("squad")
    out["tilt"] = out["tilt"].where(out["tilt"].notna(), out["poss"] / 100.0)
    return out


def parse_keepersadv(dest: Path) -> pd.DataFrame:
    path = dest / "keepersadv.html"
    if not path.exists():
        return pd.DataFrame()
    html = path.read_text(encoding="utf-8")
    rows = [r for r in _stat_rows(html, "stats_keeper_adv") if r.get("player") and r.get("player") != "Player"]
    if rows:
        nineties = _num(rows, "minutes_90s")
        minutes = _num(rows, "minutes")
        minutes = minutes.where(minutes.notna(), nineties * 90.0)
        psxg = _num(rows, "gk_psxg")
        psxg_net = _num(rows, "gk_psxg_net")
        psxg_net90 = _num(rows, "gk_psxg_net_per90")
        psxg_net90 = psxg_net90.where(psxg_net90.notna(), psxg_net / nineties.replace(0, np.nan))
        out = pd.DataFrame(
            {
                "player": [r.get("player", "") for r in rows],
                "squad": [r.get("team") or r.get("squad") or "" for r in rows],
                "minutes_gk": minutes,
                "psxg": psxg,
                "psxg_net": psxg_net,
                "psxg_net90": psxg_net90,
            }
        )
        return out.drop_duplicates("player")
    gk = _pick_table(_tables_from_html(html), {"player", "psxg"})
    if gk is None or not any("player" in str(c).lower() for c in gk.columns):
        return pd.DataFrame()
    pc = next(c for c in gk.columns if "player" in str(c).lower())
    sc = _squad_col(gk)
    nineties = _col(gk, "90s")
    minutes = _col(gk, "Min", "Minutes")
    minutes = minutes.where(minutes.notna(), nineties * 90.0)
    psxg = _col(gk, "Expected_PSxG", "PSxG")
    psxg_net = _col(gk, "Expected_PSxG+/-", "PSxG+/-")
    psxg_net90 = _col(gk, "Expected_/90")
    psxg_net90 = psxg_net90.where(psxg_net90.notna(), psxg_net / nineties.replace(0, np.nan))
    out = pd.DataFrame(
        {
            "player": gk[pc].astype(str),
            "squad": gk[sc].astype(str) if sc else "",
            "minutes_gk": minutes,
            "psxg": psxg,
            "psxg_net": psxg_net,
            "psxg_net90": psxg_net90,
        }
    )
    return out.loc[~out["player"].str.contains("Player", na=False)].drop_duplicates("player")


def parse_fbref_folder(dest: Path) -> pd.DataFrame:
    stats_html = (dest / "stats.html").read_text(encoding="utf-8") if (dest / "stats.html").exists() else ""
    def_html = (dest / "defense.html").read_text(encoding="utf-8") if (dest / "defense.html").exists() else ""
    st = _pick_table(_tables_from_html(stats_html), {"player", "squad", "min", "sh"}) if stats_html else None
    de = _pick_table(_tables_from_html(def_html), {"player", "tkl", "int"}) if def_html else None
    gk = parse_keepersadv(dest)
    if st is None and de is None and gk.empty:
        return pd.DataFrame()
    if st is not None:
        base = st
    elif de is not None:
        base = de
    else:
        base = None
    if base is not None:
        player_col = next(c for c in base.columns if "player" in str(c).lower())
        squad_col = next((c for c in base.columns if "squad" in str(c).lower()), None)
        out = pd.DataFrame({"player": base[player_col].astype(str), "squad": base[squad_col].astype(str) if squad_col else ""})
        out = out.loc[~out["player"].str.contains("Player", na=False)]
    else:
        out = gk[["player", "squad"]].copy()
    if st is not None:
        st = st.copy()
        st["_p"] = st[next(c for c in st.columns if "player" in str(c).lower())].astype(str)
        out = out.merge(
            pd.DataFrame(
                {
                    "player": st["_p"],
                    "minutes": _col(st, "Min", "Minutes"),
                    "shots": _col(st, "Sh", "Shots"),
                    "shots_on_target": _col(st, "SoT"),
                }
            ),
            on="player",
            how="left",
        )
    if de is not None:
        de = de.copy()
        de["_p"] = de[next(c for c in de.columns if "player" in str(c).lower())].astype(str)
        out = out.merge(
            pd.DataFrame(
                {
                    "player": de["_p"],
                    "tkl": _col(de, "Tkl"),
                    "interceptions": _col(de, "Int"),
                    "blocks": _col(de, "Blocks"),
                    "clearances": _col(de, "Clr"),
                }
            ),
            on="player",
            how="left",
        )
    if not gk.empty:
        out = out.merge(gk[["player", "psxg", "psxg_net", "psxg_net90", "minutes_gk"]], on="player", how="left")
        if "minutes" not in out.columns:
            out["minutes"] = out["minutes_gk"]
        else:
            out["minutes"] = out["minutes"].fillna(out["minutes_gk"])
        out = out.drop(columns=["minutes_gk"], errors="ignore")
        extra_gk = gk.loc[~gk["player"].isin(out["player"])]
        if not extra_gk.empty:
            extra = extra_gk.rename(columns={"minutes_gk": "minutes"})
            out = pd.concat([out, extra], ignore_index=True)
    return out.drop_duplicates("player")


def map_fbref_to_fpl(fbref: pd.DataFrame, fpl: pd.DataFrame) -> pd.DataFrame:
    fpl = fpl.copy()
    fpl["nk"] = (fpl["web_name"].map(normalize_name) + "|" + fpl.get("second_name", pd.Series("", index=fpl.index)).map(normalize_name)).str.strip("|")
    fpl["full"] = (
        fpl.get("first_name", pd.Series("", index=fpl.index)).fillna("").map(normalize_name)
        + " "
        + fpl.get("second_name", pd.Series("", index=fpl.index)).fillna("").map(normalize_name)
    ).str.strip()
    fpl["last"] = fpl["full"].map(last_name)
    rows = []
    used = set()
    for _, r in fbref.iterrows():
        n = normalize_name(r["player"])
        last = last_name(n)
        squad = team_key(r.get("squad"))
        hit = fpl.loc[fpl["full"] == n]
        method = "full"
        if hit.empty:
            hit = fpl.loc[fpl["web_name"].map(normalize_name) == n]
            method = "web"
        if len(hit) != 1:
            cand = fpl.loc[fpl["last"] == last]
            if squad and "team_key" in fpl.columns:
                cand2 = cand.loc[cand["team_key"] == squad]
                if len(cand2) == 1:
                    hit, method = cand2, "last+team"
                elif len(cand) == 1:
                    hit, method = cand, "last"
            elif len(cand) == 1:
                hit, method = cand, "last"
        if len(hit) == 1:
            fid = int(hit["fpl_id"].iloc[0])
            if fid in used:
                continue
            used.add(fid)
            rows.append({"fpl_id": fid, "fbref_player": r["player"], "squad": r.get("squad"), "match_method": method})
    return pd.DataFrame(rows)


def load_fbref(season: str, settings: Settings | None = None, force: bool = False) -> int:
    settings = settings or load_settings()
    dest = download_fbref(season, settings, force=force)
    parsed = parse_fbref_folder(dest)
    conn = connect(settings)
    init_db(conn, settings)
    conn.execute("DELETE FROM fact_fbref_player WHERE season = ?", (season,))
    conn.execute("DELETE FROM map_fbref WHERE season = ?", (season,))
    try:
        conn.execute("DELETE FROM fact_fbref_squad WHERE season = ?", (season,))
    except sqlite3.OperationalError:
        pass
    squad = parse_squad_possession(dest)
    if not squad.empty:
        squad = squad.copy()
        squad["season"] = season
        _to_sql(conn, squad[["season", "squad", "poss", "att3rd", "def3rd", "att_pen", "tilt"]], "fact_fbref_squad")
    if parsed.empty:
        conn.commit()
        conn.close()
        print(
            f"WARNING: FBref returned 0 players for {season} from {dest}. "
            "Save squad / shooting / defense / keepersadv / possession HTML locally "
            "(Cloudflare blocks auto-fetch), then rerun: fpl-model fbref --season "
            f"{season}. DefCon priors and GK PSxG stay weak without it."
        )
        return 0
    parsed = parsed.copy()
    parsed["season"] = season
    cols = ["minutes", "shots", "shots_on_target", "tkl", "interceptions", "blocks", "clearances", "psxg", "psxg_net", "psxg_net90"]
    for c in cols:
        if c not in parsed.columns:
            parsed[c] = pd.NA
    _to_sql(conn, parsed[["season", "player", "squad"] + cols], "fact_fbref_player")
    fpl = pd.read_sql_query(
        """
        SELECT p.fpl_id, p.web_name, p.first_name, p.second_name, t.name AS team_name
        FROM dim_player p
        LEFT JOIN dim_team t ON t.season = p.season AND t.team_id = p.team_id
        WHERE p.season = ?
        """,
        conn,
        params=(season,),
    )
    if not fpl.empty:
        fpl["team_key"] = fpl["team_name"].map(team_key)
        mapped = map_fbref_to_fpl(parsed, fpl)
        if not mapped.empty:
            mapped["season"] = season
            _to_sql(conn, mapped, "map_fbref")
            n = len(mapped)
        else:
            n = 0
    else:
        n = 0
    conn.commit()
    conn.close()
    if n == 0:
        print(
            f"WARNING: FBref parsed rows but mapped 0/{len(fpl) if not fpl.empty else 0} "
            f"FPL players for {season}. Check name/team matching or re-save tables."
        )
    if "psxg" in parsed.columns and parsed["psxg"].notna().sum() == 0:
        print(
            "FBref keepersadv HTML has empty PSxG cells (common when the page is saved "
            "before Opta columns load). Team tilt will use possession % instead of Att 3rd. "
            "Re-save keepersadv/possession after the full table appears, then rerun fbref."
        )
    return int(n)


def attach_fbref(conn, season: str, out: pd.DataFrame) -> pd.DataFrame:
    """Join mapped FBref rates onto an as-of feature frame (index = fpl_id)."""
    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='map_fbref'"
    ).fetchone()
    fb = pd.DataFrame()
    if exists:
        try:
            fb = pd.read_sql_query(
                """
                SELECT m.fpl_id, f.minutes AS fbref_minutes, f.shots, f.shots_on_target,
                       f.tkl, f.interceptions, f.blocks, f.clearances,
                       f.psxg, f.psxg_net, f.psxg_net90
                FROM map_fbref m
                JOIN fact_fbref_player f
                  ON f.season = m.season AND f.player = m.fbref_player
                WHERE m.season = ?
                """,
                conn,
                params=(season,),
            )
        except Exception:
            fb = pd.read_sql_query(
                """
                SELECT m.fpl_id, f.minutes AS fbref_minutes, f.shots, f.shots_on_target,
                       f.tkl, f.interceptions, f.blocks, f.clearances
                FROM map_fbref m
                JOIN fact_fbref_player f
                  ON f.season = m.season AND f.player = m.fbref_player
                WHERE m.season = ?
                """,
                conn,
                params=(season,),
            )
    if not fb.empty:
        fb = fb.drop_duplicates("fpl_id").set_index("fpl_id")
        cbit = fb[["tkl", "interceptions", "blocks", "clearances"]].sum(axis=1, min_count=1)
        mins = fb["fbref_minutes"].fillna(0)
        extra = pd.DataFrame(index=fb.index)
        extra["fbref_cbit90"] = np.where(mins > 0, cbit / mins * 90.0, np.nan)
        extra["fbref_shots90"] = np.where(mins > 0, fb["shots"].fillna(0) / mins * 90.0, np.nan)
        if "psxg_net90" in fb.columns:
            extra["psxg_net90"] = pd.to_numeric(fb["psxg_net90"], errors="coerce")
        extra["fbref_matched"] = 1
        out = out.join(extra)
    if "fbref_matched" not in out.columns:
        out["fbref_matched"] = 0
    else:
        out["fbref_matched"] = out["fbref_matched"].fillna(0).astype(int)
    squad_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_fbref_squad'"
    ).fetchone()
    if squad_exists:
        sq = pd.read_sql_query(
            "SELECT squad, tilt FROM fact_fbref_squad WHERE season = ?",
            conn,
            params=(season,),
        )
        teams = pd.read_sql_query(
            "SELECT team_id, name FROM dim_team WHERE season = ?",
            conn,
            params=(season,),
        )
        if not sq.empty and not teams.empty and "team_id" in out.columns:
            sq["team_key"] = sq["squad"].map(team_key)
            teams["team_key"] = teams["name"].map(team_key)
            tilt_by_id = teams.merge(sq[["team_key", "tilt"]], on="team_key", how="inner").drop_duplicates("team_id")
            tilt_map = tilt_by_id.set_index("team_id")["tilt"]
            out["team_tilt"] = pd.to_numeric(out["team_id"].map(tilt_map), errors="coerce")
    return out
