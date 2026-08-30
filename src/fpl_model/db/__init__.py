from __future__ import annotations

import sqlite3
from pathlib import Path

from fpl_model.config import Settings, load_settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(settings: Settings | None = None) -> sqlite3.Connection:
    settings = settings or load_settings()
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection | None = None, settings: Settings | None = None) -> sqlite3.Connection:
    close = False
    if conn is None:
        conn = connect(settings)
        close = False
    schema = SCHEMA_PATH.read_text()
    conn.executescript(schema)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(map_player_ids)").fetchall()}
    if "match_method" not in cols:
        conn.execute("ALTER TABLE map_player_ids ADD COLUMN match_method TEXT")
    conn.commit()
    return conn
