from __future__ import annotations

import numpy as np
import pandas as pd

# How many of this position typically start. Rank beyond this is a rotation risk.
XI_SLOTS = {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2}
MIN_RIVAL_GAMES = 3


def competing_start_rates(past: pd.DataFrame) -> pd.DataFrame:
    """Start rate when the main teammate in the same position also started.

    Also ranks last-5 starts within team+position vs typical XI slots.
    Index is ``fpl_id``.
    """
    need = {"fpl_id", "team_id", "position", "round", "starts"}
    empty = pd.DataFrame(
        columns=["rotation_start_pct", "rival_games", "pos_slot_rank", "slot_security"]
    )
    if past is None or past.empty or not need.issubset(past.columns):
        return empty

    g = past.loc[past["team_id"].notna() & past["position"].notna(), list(need)].copy()
    if g.empty:
        return empty
    g["fpl_id"] = pd.to_numeric(g["fpl_id"], errors="coerce")
    g["round"] = pd.to_numeric(g["round"], errors="coerce")
    g = g.dropna(subset=["fpl_id", "round"])
    g["fpl_id"] = g["fpl_id"].astype(int)
    g["started"] = pd.to_numeric(g["starts"], errors="coerce").fillna(0) >= 1
    tot_starts = g.groupby("fpl_id")["starts"].sum()
    meta = g.groupby("fpl_id").agg(team_id=("team_id", "first"), position=("position", "first"))

    rival_of: dict[int, int | None] = {}
    for (_, _), grp in meta.groupby(["team_id", "position"]):
        members = [int(x) for x in grp.index.tolist()]
        for pid in members:
            others = [o for o in members if o != pid]
            if not others:
                rival_of[pid] = None
                continue
            rival_of[pid] = max(others, key=lambda o: float(tot_starts.get(o, 0)))

    started_rounds = (
        g.loc[g["started"], ["fpl_id", "round"]].groupby("fpl_id")["round"].apply(set).to_dict()
    )
    by_player = {int(pid): sub for pid, sub in g.groupby("fpl_id")}

    rot_rows: list[dict] = []
    for pid, rival in rival_of.items():
        mine = by_player.get(pid)
        own = float(mine["started"].mean()) if mine is not None and len(mine) else np.nan
        if rival is None or mine is None:
            rot_rows.append({"fpl_id": pid, "rotation_start_pct": own, "rival_games": 0})
            continue
        rival_set = started_rounds.get(rival, set())
        n_r = len(rival_set)
        if n_r < MIN_RIVAL_GAMES:
            rot_rows.append({"fpl_id": pid, "rotation_start_pct": own, "rival_games": n_r})
            continue
        on_rival = mine.loc[mine["round"].isin(rival_set), "started"]
        rot_rows.append(
            {
                "fpl_id": pid,
                "rotation_start_pct": float(on_rival.mean()) if len(on_rival) else np.nan,
                "rival_games": n_r,
            }
        )
    rot = pd.DataFrame(rot_rows).set_index("fpl_id") if rot_rows else empty.copy()

    recent = sorted(g["round"].dropna().unique())[-5:]
    l5 = g.loc[g["round"].isin(recent)].groupby("fpl_id")["started"].sum()
    meta = meta.copy()
    meta["l5"] = l5.reindex(meta.index).fillna(0)
    meta["pos_slot_rank"] = meta.groupby(["team_id", "position"])["l5"].rank(ascending=False, method="min")
    slots = meta["position"].map(XI_SLOTS).fillna(4).astype(float)
    meta["slot_security"] = (slots / meta["pos_slot_rank"].clip(lower=1)).clip(0.40, 1.0)

    out = meta[["pos_slot_rank", "slot_security"]].join(rot, how="left")
    if "rival_games" not in out.columns:
        out["rival_games"] = 0
    else:
        out["rival_games"] = out["rival_games"].fillna(0)
    if "rotation_start_pct" not in out.columns:
        out["rotation_start_pct"] = np.nan
    return out[["rotation_start_pct", "rival_games", "pos_slot_rank", "slot_security"]]
