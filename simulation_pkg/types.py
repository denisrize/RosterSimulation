from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class RaceContext:
    cluster: str
    classification: str
    date: str
    distance: Optional[float] = None
    verticalMeters: Optional[float] = None


@dataclass
class SimulationPaths:
    model_path: str
    rider_features_path: str
    trueskill_leader_path: str
    trueskill_team_path: str
    hyperparams_path: Optional[str] = None
    feature_columns_path: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    clusters: Optional[List[str]] = None
    leader_feature_columns: Optional[List[str]] = None
    teammate_feature_columns: Optional[List[str]] = None


@dataclass
class SimulationRunConfig:
    team_name: str
    race_name: str
    year: int
    level: str
    scheme: str
    pool_size: int
    roster_size: int
    top_k: int
    output_dir: str
    race_context: RaceContext
    time_gap: Optional[int] = None


@dataclass
class SimulationConfig:
    paths: SimulationPaths
    run: SimulationRunConfig


def require_keys(data: Dict[str, Any], keys: List[str], label: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"Missing required keys in {label}: {', '.join(missing)}")
