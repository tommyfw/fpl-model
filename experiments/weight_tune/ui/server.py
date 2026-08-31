from __future__ import annotations

import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from fpl_model.config import load_settings

from ui import lab

STATIC = Path(__file__).resolve().parent / "static"


class LabHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, format: str, *args) -> None:
        print(f"lab-ui {self.address_string()} {format % args}")

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
        if path == "/api/metrics":
            self._json(lab.metrics())
            return
        if path in ("/", "/index.html"):
            self.path = "/index.html"
        super().do_GET()

    def _meta(self) -> dict:
        settings = load_settings()
        snaps = lab.snapshot_list()
        mixes = lab.mix_labels()
        default = snaps[0] if snaps else None
        return {
            "lab": True,
            "snapshots": snaps,
            "default": default,
            "mixes": mixes,
            "default_mix": (
                "yaml_as_is__yaml_as_is__yaml_as_is"
                if "yaml_as_is__yaml_as_is__yaml_as_is" in mixes
                else (mixes[0] if mixes else "yaml_as_is__yaml_as_is__yaml_as_is")
            ),
            "horizon_gws": settings.horizon_gws,
        }

    def _scope(self, qs: dict[str, str]) -> tuple[str | None, int, str]:
        snaps = lab.snapshot_list()
        mixes = lab.mix_labels()
        season = qs.get("season") or (snaps[0]["season"] if snaps else None)
        as_of = int(qs["as_of"]) if qs.get("as_of") else (snaps[0]["as_of_gw"] if snaps else 0)
        mix = qs.get("mix") or (mixes[0] if mixes else "yaml_as_is__yaml_as_is__yaml_as_is")
        return season, as_of, mix

    def _players(self, qs: dict[str, str]) -> dict:
        season, as_of, mix = self._scope(qs)
        if season is None:
            return {"players": [], "error": "no feature caches — run python run.py first"}
        players = lab.list_players(
            season,
            as_of,
            mix,
            position=qs.get("position") or None,
            q=qs.get("q") or None,
            sort=qs.get("sort") or "overall",
        )
        return {"season": season, "as_of_gw": as_of, "mix": mix, "players": players}

    def _player(self, qs: dict[str, str]) -> dict | None:
        season, as_of, mix = self._scope(qs)
        try:
            fpl_id = int(qs.get("id") or 0)
        except ValueError:
            fpl_id = 0
        if season is None or not fpl_id:
            self.send_error(404)
            return None
        player = lab.get_player(season, as_of, mix, fpl_id)
        if player is None:
            self.send_error(404)
            return None
        return player

    def _xi(self, qs: dict[str, str]) -> dict:
        season, as_of, mix = self._scope(qs)
        if season is None:
            return {"ok": False, "error": "no caches"}
        try:
            budget = float(qs.get("budget") or 100)
        except ValueError:
            budget = 100.0
        objective = qs.get("objective") or "xpts"
        return lab.xi(season, as_of, mix, budget=budget, objective=objective)

    def _json(self, payload: dict) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(host: str = "127.0.0.1", port: int = 8777) -> None:
    httpd = ThreadingHTTPServer((host, port), LabHandler)
    print(f"weight-tune lab UI → http://{host}:{port}")
    print("This is not the live player-card UI. Live stays on 8766.")
    httpd.serve_forever()
