PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS dim_season (
    season TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS dim_team (
    season TEXT NOT NULL,
    team_id INTEGER NOT NULL,
    name TEXT,
    short_name TEXT,
    PRIMARY KEY (season, team_id)
);

CREATE TABLE IF NOT EXISTS dim_player (
    season TEXT NOT NULL,
    fpl_id INTEGER NOT NULL,
    web_name TEXT,
    first_name TEXT,
    second_name TEXT,
    position TEXT,
    team_id INTEGER,
    status TEXT,
    chance_of_playing_this REAL,
    chance_of_playing_next REAL,
    news TEXT,
    penalties_order INTEGER,
    corners_order INTEGER,
    direct_fk_order INTEGER,
    now_cost INTEGER,
    code INTEGER,
    selected_by_percent REAL,
    PRIMARY KEY (season, fpl_id)
);

CREATE TABLE IF NOT EXISTS map_player_ids (
    season TEXT,
    fpl_id INTEGER,
    understat_id INTEGER,
    fpl_name TEXT,
    understat_name TEXT,
    match_method TEXT
);

CREATE TABLE IF NOT EXISTS fact_fpl_player_gw (
    season TEXT NOT NULL,
    fpl_id INTEGER NOT NULL,
    round INTEGER NOT NULL,
    fixture_id INTEGER,
    opponent_team INTEGER,
    was_home INTEGER,
    minutes INTEGER,
    starts INTEGER,
    total_points INTEGER,
    goals_scored INTEGER,
    assists INTEGER,
    bonus INTEGER,
    bps INTEGER,
    clean_sheets INTEGER,
    goals_conceded INTEGER,
    saves INTEGER,
    expected_goals REAL,
    expected_assists REAL,
    expected_goal_involvements REAL,
    expected_goals_conceded REAL,
    value INTEGER,
    selected INTEGER,
    defensive_contribution REAL,
    tackles INTEGER,
    recoveries INTEGER,
    clearances_blocks_interceptions INTEGER,
    yellow_cards INTEGER,
    red_cards INTEGER,
    penalties_missed INTEGER,
    penalties_saved INTEGER,
    threat REAL,
    creativity REAL,
    influence REAL,
    ict_index REAL,
    kickoff_time TEXT,
    PRIMARY KEY (season, fpl_id, round, fixture_id)
);

CREATE TABLE IF NOT EXISTS fact_fixture (
    season TEXT NOT NULL,
    fixture_id INTEGER NOT NULL,
    event INTEGER,
    team_h INTEGER,
    team_a INTEGER,
    team_h_score INTEGER,
    team_a_score INTEGER,
    finished INTEGER,
    kickoff_time TEXT,
    team_h_difficulty INTEGER,
    team_a_difficulty INTEGER,
    PRIMARY KEY (season, fixture_id)
);

CREATE TABLE IF NOT EXISTS feat_player_asof (
    season TEXT NOT NULL,
    as_of_gw INTEGER NOT NULL,
    fpl_id INTEGER NOT NULL,
    position TEXT,
    team_id INTEGER,
    price REAL,
    features_json TEXT NOT NULL,
    PRIMARY KEY (season, as_of_gw, fpl_id)
);

CREATE TABLE IF NOT EXISTS score_player_asof (
    season TEXT NOT NULL,
    as_of_gw INTEGER NOT NULL,
    fpl_id INTEGER NOT NULL,
    position TEXT,
    team_id INTEGER,
    web_name TEXT,
    U REAL,
    F REAL,
    R REAL,
    T REAL,
    Q REAL,
    M REAL,
    O REAL,
    V REAL,
    overall REAL,
    upside REAL,
    xpts REAL,
    action TEXT,
    card_json TEXT NOT NULL,
    PRIMARY KEY (season, as_of_gw, fpl_id)
);

CREATE TABLE IF NOT EXISTS fact_understat_player_match (
    season TEXT NOT NULL,
    understat_id INTEGER NOT NULL,
    match_id INTEGER,
    date TEXT,
    minutes REAL,
    position TEXT,
    shots REAL,
    shots_in_box REAL,
    key_passes REAL,
    goals REAL,
    assists REAL,
    xg REAL,
    xa REAL,
    npg REAL,
    npxg REAL,
    xgchain REAL,
    xgbuildup REAL,
    open_npxg REAL,
    sp_xg REAL,
    h_team TEXT,
    a_team TEXT
);

CREATE TABLE IF NOT EXISTS fact_understat_team_match (
    season TEXT NOT NULL,
    match_id INTEGER,
    datetime TEXT,
    date TEXT,
    finished INTEGER,
    team_title TEXT,
    team_key TEXT,
    understat_team_id INTEGER,
    is_home INTEGER,
    xg_for REAL,
    xg_against REAL,
    gf REAL,
    ga REAL
);

CREATE INDEX IF NOT EXISTS idx_us_player_date ON fact_understat_player_match (season, understat_id, date);
CREATE INDEX IF NOT EXISTS idx_us_team_date ON fact_understat_team_match (season, date);

CREATE TABLE IF NOT EXISTS fact_fbref_player (
    season TEXT NOT NULL,
    player TEXT NOT NULL,
    squad TEXT,
    minutes REAL,
    shots REAL,
    shots_on_target REAL,
    tkl REAL,
    interceptions REAL,
    blocks REAL,
    clearances REAL,
    psxg REAL,
    psxg_net REAL,
    psxg_net90 REAL,
    PRIMARY KEY (season, player, squad)
);

CREATE TABLE IF NOT EXISTS fact_fbref_squad (
    season TEXT NOT NULL,
    squad TEXT NOT NULL,
    poss REAL,
    att3rd REAL,
    def3rd REAL,
    att_pen REAL,
    tilt REAL,
    PRIMARY KEY (season, squad)
);

CREATE TABLE IF NOT EXISTS map_fbref (
    season TEXT,
    fpl_id INTEGER,
    fbref_player TEXT,
    squad TEXT,
    match_method TEXT
);
