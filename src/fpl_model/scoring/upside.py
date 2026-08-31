from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fpl_model.features.stats import signed_to_score, to_score_0_100


def ownership_score(overall: pd.Series, selected: pd.Series) -> pd.Series:
    """Low owned + high Overall → above 50; template + mediocre → below 50."""
    sel = to_score_0_100(pd.to_numeric(selected, errors="coerce"))
    ov = pd.to_numeric(overall, errors="coerce").fillna(50.0)
    return (50.0 + 0.5 * (ov - 50.0) * (50.0 - sel) / 50.0).clip(0.0, 100.0)


def compute_upside(df: pd.DataFrame, weights: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    # Residual = xG − goals (and xA − assists). Wasted chances → positive → Upside > 50.
    # Overperforming finishers/creators (goals > xG) → negative residual → Upside < 50.
    fin = pd.to_numeric(out.get("npxg90"), errors="coerce") - pd.to_numeric(out.get("goals90"), errors="coerce")
    cre = pd.to_numeric(out.get("npxa90"), errors="coerce") - pd.to_numeric(out.get("assists90"), errors="coerce")
    mtr = pd.to_numeric(out.get("minutes_pct_l5"), errors="coerce") - pd.to_numeric(out.get("minutes_pct"), errors="coerce")
    role_raw = pd.to_numeric(out.get("role_change_raw"), errors="coerce")
    price_chg = pd.to_numeric(out.get("price_change"), errors="coerce")
    atk_tr = pd.to_numeric(out.get("team_attack_trend"), errors="coerce")
    def_tr = pd.to_numeric(out.get("team_defence_trend"), errors="coerce")

    out["up_finishing"] = signed_to_score(fin.fillna(0), default_scale=0.12)
    out["up_creation"] = signed_to_score(cre.fillna(0), default_scale=0.10)
    out["up_minutes_trend"] = signed_to_score(mtr.fillna(0), default_scale=0.15)
    out["up_role_change"] = signed_to_score(role_raw.fillna(0), default_scale=8.0)
    out["up_price"] = signed_to_score((-price_chg).fillna(0), default_scale=0.30)

    gk = out["position"] == "GK"
    out.loc[gk, "up_finishing"] = 50.0
    out.loc[gk, "up_creation"] = 50.0

    tm = weights["team_mix"]
    team_raw = pd.Series(0.0, index=out.index)
    for pos, mix in tm.items():
        idx = out["position"] == pos
        if not idx.any():
            continue
        team_raw.loc[idx] = mix["attack"] * atk_tr.fillna(0) + mix["defence"] * def_tr.fillna(0)
    out["up_team_trend"] = signed_to_score(team_raw, default_scale=0.20)

    selected = out["last_selected"] if "last_selected" in out.columns else out.get("selected")
    out["up_ownership"] = ownership_score(out["overall"], selected if selected is not None else pd.Series(np.nan, index=out.index))

    uwu = weights["upside"]
    out["upside"] = (
        uwu["finishing"] * out["up_finishing"]
        + uwu["creation"] * out["up_creation"]
        + uwu["minutes_trend"] * out["up_minutes_trend"]
        + uwu["team_trend"] * out["up_team_trend"]
        + uwu["role_change"] * out["up_role_change"]
        + uwu["price"] * out["up_price"]
        + uwu["ownership"] * out["up_ownership"]
    )
    return out
