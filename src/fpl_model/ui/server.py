from __future__ import annotations

import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from fpl_model.config import Settings, load_settings
from fpl_model.db import connect, init_db
from fpl_model.ui import queries

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _int_list(raw: str) -> list[int]:
    if not raw or not raw.strip():
        return []
    try:
        return [int(x) for x in raw.split(",") if x.strip()]
    except ValueError:
        return []


def _pair_list(raw: str) -> list[tuple[int, int]]:
    if not raw or not raw.strip():
        return []
    out: list[tuple[int, int]] = []
    for part in raw.split(","):
        bit = part.strip()
        if not bit:
            continue
        sep = "-" if "-" in bit else ":"
        if sep not in bit:
            return []
        a, b = bit.split(sep, 1)
        try:
            out.append((int(a), int(b)))
        except ValueError:
            return []
    return out


class UIHandler(SimpleHTTPRequestHandler):
    settings: Settings

    def __init__(self, *args, settings: Settings, **kwargs):
        self.settings = settings
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, format: str, *args) -> None:
        print(f"ui {self.address_string()} {format % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = {k: v[0] for k, v in parse_qs(parsed.query).items() if v}
        if path == "/api/meta":
            self._json(self._meta())
            return
        if path == "/api/players":
            self._json(self._players(qs))
            return
        if path == "/api/player":
            player = self._player(qs)
            if player is None:
                return
            self._json(player)
            return
        if path == "/api/xi":
            self._json(self._xi(qs))
            return
        if path == "/api/fixtures":
            self._json(self._fixtures(qs))
            return
        if path == "/api/transfers":
            self._json(self._transfers(qs))
            return
        if path == "/api/lineup":
            self._json(self._lineup(qs))
            return
        if path == "/api/rebuild":
            self._json(self._rebuild(qs))
            return
        if path == "/api/calibrate":
            self._json(self._calibrate(qs))
            return
        if path == "/api/fpl-team":
            self._json(self._fpl_team(qs))
            return
        if path in ("/", "/index.html"):
            self.path = "/index.html"
        super().do_GET()

    def _meta(self) -> dict:
        conn = connect(self.settings)
        init_db(conn, self.settings)
        snaps = queries.available_snapshots(conn)
        preferred = getattr(self.settings, "seasons", [None])[0]
        default = queries.default_snapshot(conn, preferred)
        clubs = queries.list_clubs(conn)
        conn.close()
        return {
            "snapshots": snaps,
            "default": default,
            "horizon_gws": self.settings.horizon_gws,
            "clubs": clubs,
        }

    def _players(self, qs: dict[str, str]) -> dict:
        season, as_of = self._scope(qs)
        if season is None:
            return {"players": [], "error": "no scores yet"}
        team_id = None
        raw_team = qs.get("team") or qs.get("team_id") or ""
        if raw_team.strip():
            try:
                team_id = int(raw_team)
            except ValueError:
                team_id = None
        conn = connect(self.settings)
        players = queries.list_players(
            conn,
            season,
            as_of,
            position=qs.get("position") or None,
            q=qs.get("q") or None,
            action=qs.get("action") or None,
            team_id=team_id,
            sort=qs.get("sort") or "xpts",
        )
        conn.close()
        return {"season": season, "as_of_gw": as_of, "players": players}

    def _player(self, qs: dict[str, str]) -> dict | None:
        season, as_of = self._scope(qs)
        try:
            fpl_id = int(qs.get("id") or 0)
        except ValueError:
            fpl_id = 0
        if season is None or not fpl_id:
            self.send_error(404)
            return None
        conn = connect(self.settings)
        player = queries.get_player(conn, season, as_of, fpl_id)
        conn.close()
        if player is None:
            self.send_error(404)
            return None
        return player

    def _xi(self, qs: dict[str, str]) -> dict:
        season, as_of = self._scope(qs)
        if season is None:
            return {"ok": False, "error": "no scores yet"}
        try:
            budget = float(qs.get("budget") or 100)
        except ValueError:
            budget = 100.0
        objective = qs.get("objective") or "xpts"
        if objective not in {"xpts", "overall", "combined"}:
            objective = "xpts"
        exclude = _int_list(qs.get("exclude") or "")
        exclude_teams = _int_list(qs.get("exclude_teams") or "")
        include = _int_list(qs.get("include") or "")
        lock = _int_list(qs.get("lock") or "")
        conn = connect(self.settings)
        out = queries.build_xi(
            conn,
            season,
            as_of,
            budget=budget,
            objective=objective,
            exclude=exclude,
            exclude_teams=exclude_teams,
            include=include,
            lock=lock,
        )
        conn.close()
        return out

    def _fixtures(self, qs: dict[str, str]) -> dict:
        season, as_of = self._scope(qs)
        if season is None:
            return {"teams": [], "events": [], "error": "no scores yet"}
        conn = connect(self.settings)
        out = queries.fixture_ticker(conn, season, as_of, self.settings)
        conn.close()
        return out

    def _transfers(self, qs: dict[str, str]) -> dict:
        season, as_of = self._scope(qs)
        if season is None:
            return {"ok": False, "error": "no scores yet", "suggestions": []}
        raw = qs.get("squad") or ""
        try:
            squad_ids = [int(x) for x in raw.split(",") if x.strip()]
        except ValueError:
            return {"ok": False, "error": "bad squad ids", "suggestions": []}
        try:
            bank = float(qs.get("bank") or 0)
        except ValueError:
            bank = 0.0
        try:
            budget = float(qs.get("budget") or 100)
        except ValueError:
            budget = 100.0
        try:
            ft = int(qs.get("ft") or 1)
        except ValueError:
            ft = 1
        objective = qs.get("objective") or "xpts"
        if objective not in {"xpts", "overall", "combined"}:
            objective = "xpts"
        exclude = _int_list(qs.get("exclude") or "")
        exclude_teams = _int_list(qs.get("exclude_teams") or "")
        include = _int_list(qs.get("include") or "")
        keep = _int_list(qs.get("keep") or "")
        try:
            want = int(qs.get("want") or 2)
        except ValueError:
            want = 2
        committed = _pair_list(qs.get("moves") or "")
        conn = connect(self.settings)
        out = queries.transfers_for_squad(
            conn,
            season,
            as_of,
            squad_ids,
            bank=bank,
            budget=budget,
            free_transfers=ft,
            objective=objective,
            exclude=exclude,
            exclude_teams=exclude_teams,
            include=include,
            keep=keep,
            want_transfers=want,
            committed=committed,
        )
        conn.close()
        return out

    def _lineup(self, qs: dict[str, str]) -> dict:
        season, as_of = self._scope(qs)
        if season is None:
            return {"ok": False, "error": "no scores yet", "xi": [], "bench": []}
        raw = qs.get("squad") or ""
        try:
            squad_ids = [int(x) for x in raw.split(",") if x.strip()]
        except ValueError:
            return {"ok": False, "error": "bad squad ids", "xi": [], "bench": []}
        event = None
        raw_ev = qs.get("event") or qs.get("gw") or ""
        if raw_ev.strip():
            try:
                event = int(raw_ev)
            except ValueError:
                event = None
        conn = connect(self.settings)
        try:
            return queries.lineup_for_squad(
                conn, season, as_of, squad_ids, event=event, settings=self.settings
            )
        except Exception as exc:
            return {"ok": False, "error": f"lineup failed: {exc}", "xi": [], "bench": []}
        finally:
            conn.close()

    def _rebuild(self, qs: dict[str, str]) -> dict:
        season, as_of = self._scope(qs)
        if season is None:
            return {"ok": False, "error": "no scores yet", "squad": []}
        raw = qs.get("squad") or ""
        try:
            squad_ids = [int(x) for x in raw.split(",") if x.strip()]
        except ValueError:
            return {"ok": False, "error": "bad squad ids", "squad": []}
        try:
            bank = float(qs.get("bank") or 0)
        except ValueError:
            bank = 0.0
        try:
            budget = float(qs.get("budget") or 100)
        except ValueError:
            budget = 100.0
        objective = qs.get("objective") or "xpts"
        if objective not in {"xpts", "overall", "combined"}:
            objective = "xpts"
        conn = connect(self.settings)
        out = queries.rebuild_for_squad(
            conn,
            season,
            as_of,
            squad_ids,
            chip=qs.get("chip") or "wildcard",
            bank=bank,
            budget=budget,
            objective=objective,
            exclude=_int_list(qs.get("exclude") or ""),
            exclude_teams=_int_list(qs.get("exclude_teams") or ""),
            include=_int_list(qs.get("include") or ""),
            keep=_int_list(qs.get("keep") or ""),
        )
        conn.close()
        return out

    def _calibrate(self, qs: dict[str, str]) -> dict:
        season, as_of = self._scope(qs)
        if season is None:
            return {"empty": True, "reason": "no scores yet"}
        conn = connect(self.settings)
        stored = queries.latest_calibration(conn, season, as_of - 1 if as_of > 1 else as_of)
        if not stored:
            stored = queries.latest_calibration(conn, season, None)
        conn.close()
        if stored:
            return {"empty": False, **stored}
        from fpl_model.backtest.walkforward import calibrate_asof

        target = as_of - 1 if as_of > 1 else as_of
        pack = calibrate_asof(season, target, self.settings, persist=True)
        if pack.get("empty"):
            return pack
        pos = pack.get("by_position")
        if hasattr(pos, "to_dict"):
            pack["by_position"] = pos.to_dict(orient="records")
        tab = pack.get("table")
        if hasattr(tab, "to_dict"):
            pack["table"] = tab.to_dict(orient="records") if not tab.empty else []
        return pack

    def _fpl_team(self, qs: dict[str, str]) -> dict:
        raw = qs.get("entry") or qs.get("id") or ""
        try:
            entry_id = int(raw)
        except ValueError:
            return {"ok": False, "error": "enter your FPL team ID (from the URL of your team page)"}
        from fpl_model.ingest.fpl_api import load_entry_team

        try:
            return load_entry_team(entry_id, self.settings, force=True)
        except Exception as exc:
            return {"ok": False, "error": str(exc), "ids": []}

    def _scope(self, qs: dict[str, str]) -> tuple[str | None, int]:
        conn = connect(self.settings)
        default = queries.default_snapshot(conn, qs.get("season"))
        conn.close()
        season = qs.get("season") or (default["season"] if default else None)
        as_of = int(qs["as_of"]) if qs.get("as_of") else (default["as_of_gw"] if default else 0)
        return season, as_of

    def _json(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(host: str = "127.0.0.1", port: int = 8765, settings: Settings | None = None) -> None:
    settings = settings or load_settings()
    init_db(settings=settings)
    handler = partial(UIHandler, settings=settings)
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"player cards → http://{host}:{port}")
    httpd.serve_forever()
