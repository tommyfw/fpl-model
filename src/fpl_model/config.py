from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fpl_model import CONFIG_DIR, project_root


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    db_path: Path
    vaastav_base_url: str
    seasons: list[str]
    min_as_of_gw: int
    max_as_of_gw: int
    horizon_gws: int
    min_minutes_for_rates: int
    shrink_prior_minutes: int
    defcon_threshold: dict[str, int | None]

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw" / "vaastav"


def load_settings(path: Path | None = None) -> Settings:
    raw = _load_yaml(path or CONFIG_DIR / "settings.yaml")
    root = project_root()
    data_dir = (root / raw["data_dir"]).resolve()
    db_path = (root / raw["db_path"]).resolve()
    return Settings(
        data_dir=data_dir,
        db_path=db_path,
        vaastav_base_url=raw["vaastav_base_url"].rstrip("/"),
        seasons=list(raw["seasons"]),
        min_as_of_gw=int(raw["min_as_of_gw"]),
        max_as_of_gw=int(raw["max_as_of_gw"]),
        horizon_gws=int(raw["horizon_gws"]),
        min_minutes_for_rates=int(raw["min_minutes_for_rates"]),
        shrink_prior_minutes=int(raw["shrink_prior_minutes"]),
        defcon_threshold=dict(raw["defcon_threshold"]),
    )


def load_weights(path: Path | None = None) -> dict[str, Any]:
    return _load_yaml(path or CONFIG_DIR / "weights.yaml")
