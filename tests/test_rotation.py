import pandas as pd

from fpl_model.features.rotation import competing_start_rates


def _gw(fpl_id, team, pos, rounds, starts):
    return pd.DataFrame(
        {
            "fpl_id": [fpl_id] * len(rounds),
            "team_id": [team] * len(rounds),
            "position": [pos] * len(rounds),
            "round": rounds,
            "starts": starts,
        }
    )


def test_backup_gk_rarely_starts_when_number_one_plays():
    rounds = list(range(1, 9))
    past = pd.concat(
        [
            _gw(1, 1, "GK", rounds, [1] * 8),
            _gw(2, 1, "GK", rounds, [0] * 8),
        ]
    )
    rot = competing_start_rates(past)
    assert rot.loc[1, "rotation_start_pct"] == 1.0
    assert rot.loc[2, "rotation_start_pct"] == 0.0
    assert rot.loc[1, "pos_slot_rank"] == 1
    assert rot.loc[2, "pos_slot_rank"] > 1
    assert rot.loc[1, "slot_security"] == 1.0
    assert rot.loc[2, "slot_security"] < rot.loc[1, "slot_security"]


def test_nailed_cb_pair_both_start_when_the_other_plays():
    rounds = list(range(1, 9))
    past = pd.concat(
        [
            _gw(1, 1, "DEF", rounds, [1] * 8),
            _gw(2, 1, "DEF", rounds, [1] * 8),
            _gw(3, 1, "DEF", rounds, [1] * 8),
            _gw(4, 1, "DEF", rounds, [1] * 8),
            _gw(5, 1, "DEF", rounds, [0] * 8),
        ]
    )
    rot = competing_start_rates(past)
    for pid in (1, 2, 3, 4):
        assert rot.loc[pid, "rotation_start_pct"] == 1.0
        assert rot.loc[pid, "slot_security"] == 1.0
    assert rot.loc[5, "rotation_start_pct"] == 0.0
    assert rot.loc[5, "slot_security"] < 1.0
