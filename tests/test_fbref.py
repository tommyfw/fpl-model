import pandas as pd

from fpl_model.ingest.fbref import map_fbref_to_fpl


def test_parse_commented_player_table():
    from pathlib import Path

    from fpl_model.ingest.fbref import parse_fbref_folder

    dest = Path("/tmp/fbref-test-parse")
    dest.mkdir(exist_ok=True)
    (dest / "stats.html").write_text(
        """<html><body><!--
        <table id="stats_standard">
        <thead><tr><th>Player</th><th>Squad</th><th>Min</th><th>Sh</th><th>SoT</th></tr></thead>
        <tbody><tr><td>Erling Haaland</td><td>Manchester City</td><td>90</td><td>4</td><td>2</td></tr></tbody>
        </table>
        --></body></html>""",
        encoding="utf-8",
    )
    (dest / "defense.html").write_text(
        """<html><body>
        <table id="stats_defense">
        <thead><tr><th>Player</th><th>Tkl</th><th>Int</th><th>Blocks</th><th>Clr</th></tr></thead>
        <tbody><tr><td>Erling Haaland</td><td>1</td><td>0</td><td>0</td><td>1</td></tr></tbody>
        </table>
        </body></html>""",
        encoding="utf-8",
    )
    parsed = parse_fbref_folder(dest)
    assert "Erling Haaland" in set(parsed["player"])
    assert float(parsed.loc[parsed["player"] == "Erling Haaland", "shots"].iloc[0]) == 4
    fbref = pd.DataFrame({"player": ["Erling Haaland"], "squad": ["Manchester City"]})
    fpl = pd.DataFrame(
        {
            "fpl_id": [1],
            "web_name": ["Haaland"],
            "first_name": ["Erling"],
            "second_name": ["Haaland"],
            "team_key": ["mancity"],
        }
    )
    m = map_fbref_to_fpl(fbref, fpl)
    assert len(m) == 1
    assert int(m["fpl_id"].iloc[0]) == 1
