from __future__ import annotations

import sqlite3
from pathlib import Path

from fpl_model.config import Settings, load_settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(settings: Settings | None = None) -> sqlite3.Connection:
    settings = settings or load_settings()
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path, timeout=8.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db(conn: sqlite3.Connection | None = None, settings: Settings | None = None) -> sqlite3.Connection:
    close = False
    if conn is None:
        conn = connect(settings)
        close = False
    schema = SCHEMA_PATH.read_text()
    conn.executescript(schema)
    def _ensure(table: str, col: str, spec: str) -> None:
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if col not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {spec}")

    _ensure("map_player_ids", "match_method", "TEXT")
    _ensure("dim_player", "code", "INTEGER")
    _ensure("dim_player", "selected_by_percent", "REAL")
    _ensure("dim_player", "news", "TEXT")
    _ensure("fact_fpl_player_gw", "xp", "REAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dim_asof_snapshot (
            season TEXT NOT NULL,
            as_of_gw INTEGER NOT NULL,
            is_beta INTEGER NOT NULL DEFAULT 0,
            official_as_of INTEGER,
            PRIMARY KEY (season, as_of_gw)
        )
        """
    )
    _ensure("fact_understat_player_match", "open_npxg", "REAL")
    _ensure("fact_understat_player_match", "sp_xg", "REAL")
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "fact_fbref_player" in tables:
        for col in ("psxg", "psxg_net", "psxg_net90"):
            _ensure("fact_fbref_player", col, "REAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_calibrate_asof (
            season TEXT NOT NULL,
            as_of_gw INTEGER NOT NULL,
            n INTEGER,
            mae REAL,
            bias REAL,
            unmatched INTEGER,
            breakdown_json TEXT,
            computed_at TEXT,
            PRIMARY KEY (season, as_of_gw)
        )
        """
    )
    conn.commit()
    return conn
