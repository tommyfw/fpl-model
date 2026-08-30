from __future__ import annotations

import numpy as np
import pandas as pd


def per90(total: pd.Series, minutes: pd.Series) -> pd.Series:
    minutes = minutes.clip(lower=0)
    return np.where(minutes > 0, total / minutes * 90.0, 0.0)


def shrink_to_mean(rate: pd.Series, minutes: pd.Series, prior_mean: float, prior_minutes: int) -> pd.Series:
    m = minutes.clip(lower=0).astype(float)
    w = m / (m + prior_minutes)
    return w * rate + (1 - w) * prior_mean


def percentile_0_100(values: pd.Series) -> pd.Series:
    valid = values.dropna()
    if valid.empty:
        return pd.Series(50.0, index=values.index)
    ranks = values.rank(method="average", pct=True)
    return ranks * 100.0


def to_score_0_100(values: pd.Series, invert: bool = False) -> pd.Series:
    s = percentile_0_100(values)
    if invert:
        s = 100.0 - s
    return s.fillna(50.0)
