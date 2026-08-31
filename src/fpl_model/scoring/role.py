from __future__ import annotations

import pandas as pd

# Understat codes → attacking-role score (0–100). Used for Role.pitch_position.
PITCH_ATTACK = {
    "FW": 95,
    "FWL": 88,
    "FWR": 88,
    "AMC": 92,
    "AML": 86,
    "AMR": 85,
    "MC": 76,
    "ML": 74,
    "MR": 74,
    "DMC": 40,
    "DML": 52,
    "DMR": 52,
    "DL": 72,
    "DR": 72,
    "DC": 30,
    "GK": 12,
}

# Central / advanced (not wide).
PITCH_CENTRAL = {
    "FW": 78,
    "AMC": 95,
    "MC": 90,
    "DMC": 88,
    "DC": 70,
    "AML": 42,
    "AMR": 40,
    "ML": 32,
    "MR": 32,
    "DL": 22,
    "DR": 22,
    "FWL": 45,
    "FWR": 40,
    "DML": 50,
    "DMR": 50,
    "GK": 20,
}


def pitch_attack_score(pos: str | None) -> float:
    if pos is None or (isinstance(pos, float) and pd.isna(pos)):
        return 50.0
    return float(PITCH_ATTACK.get(str(pos).upper(), 50.0))


def pitch_central_score(pos: str | None) -> float:
    if pos is None or (isinstance(pos, float) and pd.isna(pos)):
        return 50.0
    return float(PITCH_CENTRAL.get(str(pos).upper(), 50.0))


def set_piece_score(corners_order, dfk_order) -> float:
    orders = []
    for o in (corners_order, dfk_order):
        if pd.notna(o) and float(o) > 0:
            orders.append(float(o))
    if not orders:
        return 40.0
    best = min(orders)
    if best <= 1:
        return 100.0
    if best <= 2:
        return 82.0
    if best <= 3:
        return 65.0
    return 45.0


def penalty_score(penalties_order) -> float:
    if pd.isna(penalties_order):
        return 38.0
    o = float(penalties_order)
    if o <= 1:
        return 100.0
    if o <= 2:
        return 70.0
    if o <= 3:
        return 52.0
    return 40.0


def defcon_count(row: pd.Series) -> float:
    dc = row.get("defensive_contribution")
    if pd.notna(dc):
        return float(dc)
    cbit = float(pd.to_numeric(row.get("clearances_blocks_interceptions"), errors="coerce") or 0)
    cbit += float(pd.to_numeric(row.get("tackles"), errors="coerce") or 0)
    pos = row.get("position")
    if pos in {"MID", "FWD"}:
        cbit += float(pd.to_numeric(row.get("recoveries"), errors="coerce") or 0)
    return cbit
