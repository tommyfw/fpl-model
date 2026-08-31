from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

from fpl_model import CONFIG_DIR

TEAM_ALIASES = {
    "manchester city": "mancity",
    "man city": "mancity",
    "manchester united": "manutd",
    "man utd": "manutd",
    "man united": "manutd",
    "tottenham hotspur": "tottenham",
    "tottenham": "tottenham",
    "spurs": "tottenham",
    "nottingham forest": "forest",
    "nottm forest": "forest",
    "nott m forest": "forest",
    "wolverhampton wanderers": "wolves",
    "wolves": "wolves",
    "brighton and hove albion": "brighton",
    "brighton": "brighton",
    "newcastle united": "newcastle",
    "newcastle": "newcastle",
    "west ham united": "westham",
    "west ham": "westham",
    "wolverhampton": "wolves",
    "sheffield united": "sheffield",
    "sheffield utd": "sheffield",
    "leicester city": "leicester",
    "ipswich town": "ipswich",
    "ipswich": "ipswich",
    "leeds united": "leeds",
    "leeds": "leeds",
    "sunderland": "sunderland",
    "burnley": "burnley",
    "afc bournemouth": "bournemouth",
    "bournemouth": "bournemouth",
    "crystal palace": "palace",
    "aston villa": "villa",
    "hull city": "hull",
    "hull": "hull",
    "coventry city": "coventry",
    "coventry": "coventry",
}


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalize_name(text: str | None) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = strip_accents(str(text)).lower()
    s = s.replace("ı", "i").replace("æ", "ae").replace("ø", "o")
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def team_key(text: str | None) -> str:
    s = normalize_name(text)
    if s in TEAM_ALIASES:
        return TEAM_ALIASES[s]
    for alias, key in TEAM_ALIASES.items():
        if alias in s or s in alias:
            return key
    # "Hull City" / "Coventry City" → hull / coventry (matches Understat titles)
    s = re.sub(r"\b(city|town|united|afc)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s in TEAM_ALIASES:
        return TEAM_ALIASES[s]
    return s.replace(" ", "")


def last_name(normalized_full: str) -> str:
    parts = [p for p in normalized_full.split() if p not in {"al", "el", "de", "da", "dos", "van", "von", "bin"}]
    return parts[-1] if parts else ""


def first_name(normalized_full: str) -> str:
    parts = normalized_full.split()
    return parts[0] if parts else ""


def load_aliases(path: Path | None = None) -> list[dict]:
    p = path or (CONFIG_DIR / "player_aliases.yaml")
    if not p.exists():
        return []
    raw = yaml.safe_load(p.read_text()) or {}
    return list(raw.get("aliases") or [])


def map_players(fpl: pd.DataFrame, understat: pd.DataFrame, aliases: list[dict] | None = None) -> pd.DataFrame:
    """Match FPL rows to Understat players.

    Expected FPL columns: fpl_id, first_name, second_name, web_name, team_name
    Expected Understat columns: understat_id, player_name, team_title
    """
    fpl = fpl.copy()
    us = understat.copy()
    fpl["full"] = (fpl["first_name"].fillna("").astype(str) + " " + fpl["second_name"].fillna("").astype(str)).map(normalize_name)
    fpl["web"] = fpl["web_name"].map(normalize_name)
    fpl["last"] = fpl["full"].map(last_name)
    fpl["first"] = fpl["full"].map(first_name)
    fpl["tkey"] = fpl["team_name"].map(team_key)

    us["full"] = us["player_name"].map(normalize_name)
    us["last"] = us["full"].map(last_name)
    us["first"] = us["full"].map(first_name)
    us["tkey"] = us["team_title"].map(team_key)

    used_fpl: set[int] = set()
    used_us: set[int] = set()
    rows: list[dict] = []

    def _add(fpl_row, us_row, method: str) -> None:
        fid = int(fpl_row["fpl_id"])
        uid = int(us_row["understat_id"])
        if fid in used_fpl or uid in used_us:
            return
        used_fpl.add(fid)
        used_us.add(uid)
        rows.append(
            {
                "fpl_id": fid,
                "understat_id": uid,
                "fpl_name": f"{fpl_row.get('first_name', '')} {fpl_row.get('second_name', '')}".strip(),
                "understat_name": us_row["player_name"],
                "match_method": method,
            }
        )

    if aliases is None:
        aliases = load_aliases()
    fpl_by_id = {int(r["fpl_id"]): r for _, r in fpl.iterrows()}
    us_by_id = {int(u["understat_id"]): u for _, u in us.iterrows()}
    us_by_name = {normalize_name(u["player_name"]): u for _, u in us.iterrows()}
    for a in aliases:
        uid = a.get("understat_id")
        fid = a.get("fpl_id")
        if uid is None:
            continue
        urow = us_by_id.get(int(uid))
        if urow is None and a.get("understat_name"):
            urow = us_by_name.get(normalize_name(a["understat_name"]))
        frow = fpl_by_id.get(int(fid)) if fid is not None else None
        alias_name = normalize_name(a.get("fpl_name") or "")
        if frow is not None and alias_name:
            blob = f"{frow['full']} {frow['web']}"
            if alias_name not in blob and frow["full"] not in alias_name and frow["last"] != last_name(alias_name):
                frow = None
        if frow is None and alias_name:
            cands = [
                r
                for _, r in fpl.iterrows()
                if int(r["fpl_id"]) not in used_fpl
                and (
                    alias_name in r["full"]
                    or r["full"] in alias_name
                    or r["web"] == last_name(alias_name)
                    or r["last"] == last_name(alias_name)
                )
            ]
            if len(cands) > 1:
                cands = [c for c in cands if c["last"] == last_name(alias_name)]
            if len(cands) == 1:
                frow = cands[0]
        if urow is not None and frow is not None:
            _add(frow, urow, "manual_alias")

    fpl_by_full = defaultdict(list)
    for _, r in fpl.iterrows():
        if r["full"]:
            fpl_by_full[r["full"]].append(r)

    # 1. unique exact full name
    for _, u in us.iterrows():
        cands = fpl_by_full.get(u["full"], [])
        cands = [c for c in cands if int(c["fpl_id"]) not in used_fpl]
        if len(cands) == 1:
            _add(cands[0], u, "exact_name")
        elif len(cands) > 1:
            same_team = [c for c in cands if c["tkey"] == u["tkey"]]
            if len(same_team) == 1:
                _add(same_team[0], u, "exact_name_team")

    # 1b. unique containment (FPL legal names vs Understat common names)
    for _, u in us.iterrows():
        if int(u["understat_id"]) in used_us or not u["full"] or len(u["full"]) < 6:
            continue
        cands = []
        for _, r in fpl.iterrows():
            if int(r["fpl_id"]) in used_fpl or not r["full"]:
                continue
            a, b = u["full"], r["full"]
            if a in b or b in a:
                cands.append(r)
        if len(cands) > 1:
            cands = [c for c in cands if c["tkey"] == u["tkey"]] or cands
        if len(cands) == 1:
            _add(cands[0], u, "name_contained")

    # 2. last name + team, unique
    fpl_by_last_team = defaultdict(list)
    for _, r in fpl.iterrows():
        if r["last"] and r["tkey"]:
            fpl_by_last_team[(r["last"], r["tkey"])].append(r)
    for _, u in us.iterrows():
        if int(u["understat_id"]) in used_us:
            continue
        cands = [c for c in fpl_by_last_team.get((u["last"], u["tkey"]), []) if int(c["fpl_id"]) not in used_fpl]
        if len(cands) == 1:
            _add(cands[0], u, "last_team")

    # 3. unique last name globally
    fpl_last_counts = fpl.groupby("last")["fpl_id"].nunique()
    us_last_counts = us.groupby("last")["understat_id"].nunique()
    for _, u in us.iterrows():
        if int(u["understat_id"]) in used_us or not u["last"]:
            continue
        if fpl_last_counts.get(u["last"], 0) == 1 and us_last_counts.get(u["last"], 0) == 1:
            cands = fpl.loc[fpl["last"] == u["last"]]
            cands = cands.loc[~cands["fpl_id"].astype(int).isin(used_fpl)]
            if len(cands) == 1:
                _add(cands.iloc[0], u, "unique_last")

    # 4. web_name equals last name + team
    fpl_by_web_team = defaultdict(list)
    for _, r in fpl.iterrows():
        if r["web"] and r["tkey"]:
            fpl_by_web_team[(r["web"], r["tkey"])].append(r)
    for _, u in us.iterrows():
        if int(u["understat_id"]) in used_us:
            continue
        cands = [c for c in fpl_by_web_team.get((u["last"], u["tkey"]), []) if int(c["fpl_id"]) not in used_fpl]
        if len(cands) == 1:
            _add(cands[0], u, "web_last_team")

    mapped = pd.DataFrame(rows)
    leftover_fpl = fpl.loc[~fpl["fpl_id"].astype(int).isin(used_fpl)]
    leftover_us = us.loc[~us["understat_id"].astype(int).isin(used_us)]
    extra = []
    for _, r in leftover_fpl.iterrows():
        extra.append(
            {
                "fpl_id": int(r["fpl_id"]),
                "understat_id": None,
                "fpl_name": f"{r.get('first_name', '')} {r.get('second_name', '')}".strip(),
                "understat_name": None,
                "match_method": "unmatched_fpl",
            }
        )
    for _, u in leftover_us.iterrows():
        extra.append(
            {
                "fpl_id": None,
                "understat_id": int(u["understat_id"]),
                "fpl_name": None,
                "understat_name": u["player_name"],
                "match_method": "unmatched_understat",
            }
        )
    out = pd.concat([mapped, pd.DataFrame(extra)], ignore_index=True) if extra else mapped
    return out
