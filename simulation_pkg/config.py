from __future__ import annotations

import json
import os
from typing import Any, Dict

from .types import (
    RaceContext,
    SimulationConfig,
    SimulationPaths,
    SimulationRunConfig,
    require_keys,
)
from .io.paths import resolve_path


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError("PyYAML is required to load YAML configs") from exc
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_config(path: str) -> SimulationConfig:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.endswith((".yml", ".yaml")):
        raw = _load_yaml(path)
    else:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)

    require_keys(raw, ["paths", "run"], "config")
    base_dir = os.path.dirname(os.path.abspath(path))

    paths_data = raw["paths"]
    run_data = raw["run"]

    require_keys(
        paths_data,
        ["model_path", "rider_features_path", "trueskill_leader_path", "trueskill_team_path"],
        "paths",
    )
    require_keys(
        run_data,
        ["team_name", "race_name", "year", "level", "scheme", "pool_size", "roster_size", "top_k", "output_dir", "race_context"],
        "run",
    )

    race_context = RaceContext(**run_data["race_context"])

    paths = SimulationPaths(
        model_path=resolve_path(paths_data["model_path"], base_dir),
        hyperparams_path=resolve_path(paths_data.get("hyperparams_path"), base_dir),
        rider_features_path=resolve_path(paths_data["rider_features_path"], base_dir),
        trueskill_leader_path=resolve_path(paths_data["trueskill_leader_path"], base_dir),
        trueskill_team_path=resolve_path(paths_data["trueskill_team_path"], base_dir),
        feature_columns_path=resolve_path(paths_data.get("feature_columns_path"), base_dir),
        feature_columns=paths_data.get("feature_columns"),
        clusters=paths_data.get("clusters"),
        leader_feature_columns=paths_data.get("leader_feature_columns"),
        teammate_feature_columns=paths_data.get("teammate_feature_columns"),
    )

    run = SimulationRunConfig(
        team_name=run_data["team_name"],
        race_name=run_data["race_name"],
        year=int(run_data["year"]),
        level=run_data["level"],
        scheme=run_data["scheme"],
        pool_size=int(run_data["pool_size"]),
        roster_size=int(run_data["roster_size"]),
        top_k=int(run_data["top_k"]),
        output_dir=resolve_path(run_data["output_dir"], base_dir),
        race_context=race_context,
        time_gap=run_data.get("time_gap"),
    )

    return SimulationConfig(paths=paths, run=run)
