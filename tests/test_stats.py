import pandas as pd

from fpl_model.features.stats import per90, shrink_to_mean, to_score_0_100


def test_per90_and_shrink_towards_prior_with_few_minutes():
    rate = pd.Series(per90(pd.Series([2.0, 0.0]), pd.Series([90.0, 0.0])))
    shrunk = shrink_to_mean(rate, pd.Series([90.0, 0.0]), prior_mean=0.4, prior_minutes=450)
    assert rate.iloc[0] == 2.0
    assert shrunk.iloc[0] < 2.0
    assert shrunk.iloc[1] == 0.4


def test_scores_are_position_relative_percentiles():
    s = to_score_0_100(pd.Series([1.0, 2.0, 3.0]))
    assert list(s.round(0)) == [33.0, 67.0, 100.0]
    inv = to_score_0_100(pd.Series([1.0, 2.0, 3.0]), invert=True)
    assert list(inv.round(0)) == [67.0, 33.0, 0.0]
