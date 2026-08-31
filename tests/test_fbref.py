from pathlib import Path

import pandas as pd
import pytest

from fpl_model.ingest.fbref import map_fbref_to_fpl, parse_fbref_folder, parse_keepersadv, parse_squad_possession


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


def test_parse_keepersadv_and_possession_html():
    dest = Path(__file__).resolve().parents[1] / "data" / "raw" / "fbref" / "2026-27"
    if not (dest / "keepersadv.html").exists() or not (dest / "possession.html").exists():
        pytest.skip("FBref HTML dumps not present")
    gk = parse_keepersadv(dest)
    assert not gk.empty
    parsed = parse_fbref_folder(dest)
    assert "psxg_net90" in parsed.columns
    sq = parse_squad_possession(dest)
    assert len(sq) == 20
    # Saved FBref HTML often blanks Opta cols; possession % is the fallback tilt.
    assert sq["tilt"].notna().sum() >= 18
    assert sq["tilt"].between(0.0, 1.0).all()
    assert float(sq.loc[sq["squad"] == "Arsenal", "tilt"].iloc[0]) > 0.5


def test_parse_keepersadv_from_data_stat_html(tmp_path):
    dest = tmp_path
    (dest / "keepersadv.html").write_text(
        """<html><table id="stats_keeper_adv">
        <tr><th data-stat="player">Player</th><th data-stat="team">Squad</th>
            <th data-stat="minutes_90s">90s</th><th data-stat="gk_psxg">PSxG</th>
            <th data-stat="gk_psxg_net">PSxG+/-</th><th data-stat="gk_psxg_net_per90">/90</th></tr>
        <tr><td data-stat="player">Alisson</td><td data-stat="team">Liverpool</td>
            <td data-stat="minutes_90s">2.0</td><td data-stat="gk_psxg">3.2</td>
            <td data-stat="gk_psxg_net">0.4</td><td data-stat="gk_psxg_net_per90">0.20</td></tr>
        </table></html>""",
        encoding="utf-8",
    )
    gk = parse_keepersadv(dest)
    assert float(gk.loc[gk["player"] == "Alisson", "psxg"].iloc[0]) == 3.2
    assert float(gk.loc[gk["player"] == "Alisson", "psxg_net90"].iloc[0]) == 0.20
