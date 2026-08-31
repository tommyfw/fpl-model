import pandas as pd

from fpl_model.scoring.role import defcon_count, penalty_score, pitch_attack_score, set_piece_score


def test_pitch_attack_ranks_fw_above_cb():
    assert pitch_attack_score("FW") > pitch_attack_score("AMC") > pitch_attack_score("DC")
    assert pitch_attack_score("AMC") > pitch_attack_score("MC") > pitch_attack_score("DMC")
    assert pitch_attack_score("Sub") == 50.0


def test_penalties_and_set_pieces():
    assert penalty_score(1) == 100
    assert penalty_score(2) < penalty_score(1)
    assert set_piece_score(1, None) == 100
    assert set_piece_score(None, None) == 40


def test_defcon_prefers_official_count():
    row = pd.Series(
        {
            "defensive_contribution": 14,
            "tackles": 2,
            "clearances_blocks_interceptions": 3,
            "recoveries": 8,
            "position": "MID",
        }
    )
    assert defcon_count(row) == 14
    row2 = pd.Series(
        {
            "defensive_contribution": None,
            "tackles": 2,
            "clearances_blocks_interceptions": 3,
            "recoveries": 8,
            "position": "MID",
        }
    )
    assert defcon_count(row2) == 13
