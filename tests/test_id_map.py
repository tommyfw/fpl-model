import pandas as pd

from fpl_model.ingest.id_map import map_players, normalize_name, team_key


def test_normalize_strips_accents_and_punctuation():
    assert normalize_name("João Pedro") == "joao pedro"
    assert normalize_name("M.Salah") == "m salah"


def test_team_aliases():
    assert team_key("Man City") == team_key("Manchester City")
    assert team_key("Spurs") == team_key("Tottenham")
    assert team_key("Nott'm Forest") == team_key("Nottingham Forest")
    assert team_key("Hull City") == team_key("Hull")
    assert team_key("Coventry City") == team_key("Coventry")


def test_map_exact_and_last_team_and_unmatched():
    fpl = pd.DataFrame(
        [
            {"fpl_id": 1, "first_name": "Erling", "second_name": "Haaland", "web_name": "Haaland", "team_name": "Man City"},
            {"fpl_id": 2, "first_name": "Mohamed", "second_name": "Salah", "web_name": "Salah", "team_name": "Liverpool"},
            {"fpl_id": 3, "first_name": "John", "second_name": "Smith", "web_name": "Smith", "team_name": "Brentford"},
            {"fpl_id": 4, "first_name": "João", "second_name": "Pedro", "web_name": "J.Pedro", "team_name": "Chelsea"},
        ]
    )
    us = pd.DataFrame(
        [
            {"understat_id": 8260, "player_name": "Erling Haaland", "team_title": "Manchester City"},
            {"understat_id": 1250, "player_name": "Mohamed Salah", "team_title": "Liverpool"},
            {"understat_id": 9999, "player_name": "Joao Pedro", "team_title": "Chelsea"},
            {"understat_id": 111, "player_name": "Some Other", "team_title": "Arsenal"},
        ]
    )
    m = map_players(fpl, us)
    linked = m.dropna(subset=["fpl_id", "understat_id"])
    by_fpl = dict(zip(linked["fpl_id"].astype(int), linked["understat_id"].astype(int)))
    assert by_fpl[1] == 8260
    assert by_fpl[2] == 1250
    assert by_fpl[4] == 9999
    # legal name vs common name
    fpl2 = pd.concat(
        [
            fpl,
            pd.DataFrame(
                [
                    {
                        "fpl_id": 5,
                        "first_name": "Bruno",
                        "second_name": "Guimarães Rodriguez Moura",
                        "web_name": "Guimarães",
                        "team_name": "Newcastle",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    us2 = pd.concat(
        [
            us,
            pd.DataFrame([{"understat_id": 700, "player_name": "Bruno Guimarães", "team_title": "Newcastle United"}]),
        ],
        ignore_index=True,
    )
    m2 = map_players(fpl2, us2)
    linked2 = m2.dropna(subset=["fpl_id", "understat_id"])
    assert 5 in set(linked2["fpl_id"].astype(int))


def test_manual_alias_overrides_before_auto():
    fpl = pd.DataFrame(
        [{"fpl_id": 10, "first_name": "Rodrigo", "second_name": "Hernandez Cascante", "web_name": "Rodri", "team_name": "Man City"}]
    )
    us = pd.DataFrame(
        [{"understat_id": 2496, "player_name": "Rodri", "team_title": "Manchester City"}]
    )
    m = map_players(fpl, us, aliases=[{"understat_id": 2496, "fpl_id": 10}])
    linked = m.dropna(subset=["fpl_id", "understat_id"])
    assert int(linked.iloc[0]["understat_id"]) == 2496
    assert linked.iloc[0]["match_method"] == "manual_alias"


def test_manual_alias_uses_name_when_fpl_id_is_from_another_season():
    fpl = pd.DataFrame(
        [
            {"fpl_id": 999, "first_name": "Rodrigo", "second_name": "Hernandez Cascante", "web_name": "Rodri", "team_name": "Man City"},
            {"fpl_id": 10, "first_name": "Someone", "second_name": "Else", "web_name": "Else", "team_name": "Arsenal"},
        ]
    )
    us = pd.DataFrame([{"understat_id": 2496, "player_name": "Rodri", "team_title": "Manchester City"}])
    m = map_players(
        fpl,
        us,
        aliases=[
            {
                "understat_id": 2496,
                "fpl_id": 10,
                "fpl_name": "Rodrigo Hernandez Cascante",
                "understat_name": "Rodri",
            }
        ],
    )
    linked = m.dropna(subset=["fpl_id", "understat_id"])
    assert int(linked.iloc[0]["fpl_id"]) == 999
