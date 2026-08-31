from __future__ import annotations

import numpy as np
import pandas as pd


def per90(total: pd.Series, minutes: pd.Series) -> pd.Series:
    t = pd.to_numeric(total, errors="coerce").fillna(0).to_numpy(dtype=float)
    m = pd.to_numeric(minutes, errors="coerce").fillna(0).clip(lower=0).to_numpy(dtype=float)
    out = np.zeros(len(m), dtype=float)
    np.divide(t * 90.0, m, out=out, where=m > 0)
    idx = getattr(total, "index", None)
    return pd.Series(out, index=idx) if idx is not None else out


def shrink_to_mean(
    rate: pd.Series,
    minutes: pd.Series,
    prior_mean: float | pd.Series,
    prior_minutes: int,
    max_prior_weight: float | None = None,
) -> pd.Series:
    m = minutes.clip(lower=0).astype(float)
    w_obs = m / (m + prior_minutes)
    w_prior = 1.0 - w_obs
    if max_prior_weight is not None:
        w_prior = w_prior.clip(upper=float(max_prior_weight))
        w_obs = 1.0 - w_prior
    if isinstance(prior_mean, pd.Series):
        prior_mean = prior_mean.reindex(rate.index).fillna(0.0)
    return w_obs * rate + w_prior * prior_mean


def blend_playing_time(
    observed: pd.Series,
    minutes: pd.Series,
    prior: pd.Series,
    prior_minutes: int,
    max_prior_weight: float,
    starts: pd.Series,
    matches: pd.Series,
    grace_matches: int,
) -> pd.Series:
    """Blend last-season start/minutes share, then drop the prior once a player has
    clearly not started this season (``starts == 0`` for ``grace_matches`` team games).
    """
    m = minutes.clip(lower=0).astype(float)
    w_prior = (1.0 - m / (m + prior_minutes)).clip(upper=float(max_prior_weight))
    prior = prior.reindex(observed.index).fillna(observed)
    blended = (1.0 - w_prior) * observed + w_prior * prior
    cold = (starts.fillna(0) <= 0) & (matches.fillna(0) >= grace_matches)
    return blended.where(~cold, observed)


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


def signed_to_score(values: pd.Series, default_scale: float = 1.0) -> pd.Series:
    """Map a signed residual to 0–100 with 50 = no edge (tanh, robust scale)."""
    v = pd.to_numeric(values, errors="coerce")
    med = float(np.nanmedian(np.abs(v - np.nanmedian(v)))) if v.notna().any() else 0.0
    scale = max(med * 1.4826, float(default_scale))
    z = v.fillna(0.0) / scale
    return (50.0 + 50.0 * np.tanh(z)).clip(0.0, 100.0)
