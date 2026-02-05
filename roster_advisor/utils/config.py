"""
Configuration Module

Provides a configuration interface that automatically resolves
internal data paths based on user-provided parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any

from .data_registry import (
    DataRegistry,
    get_registry,
    RaceNotFoundError,
    TeamNotFoundError,
    SchemeNotFoundError,
    DataNotFoundError,
)
from .types import RaceContext, SimulationConfig, SimulationPaths, SimulationRunConfig


@dataclass
class UserConfig:
    """
    User-facing configuration for roster optimization.
    
    Users only need to specify these parameters:
    - team_name: The team to optimize rosters for
    - race_name: The target race
    - riders_pool: List of 4 integers for riders per category:
                   [race_cluster_leader, gc_leader, race_cluster_teammate, gc_teammate]
                   Default: [4, 4, 4, 4] = 16 riders total
    - roster_size: Number of cyclists per roster (default: 8)
    - scheme: Weighting scheme (time_lag, equal_weight, position)
    - time_horizon: Days before race date for feature cutoff (optional)
    - year: Season year (default: current year)
    - output_dir: Where to save results (default: ./results)
    - exclude_riders: Riders to exclude from pool (e.g., left the team)
    - include_riders: Riders to include in pool (e.g., new signings)
    """
    team_name: str
    race_name: str
    scheme: str = "time_lag"
    riders_pool: List[int] = field(default_factory=lambda: [4, 4, 4, 4])
    roster_size: int = 8
    year: Optional[int] = None
    time_horizon: Optional[int] = None  # Days before race
    output_dir: str = "results"
    top_k: int = 10
    level: str = "rider"
    
    # Rider inclusion/exclusion
    exclude_riders: Optional[List[str]] = None  # Riders who left the team
    include_riders: Optional[List[str]] = None  # New signings to include
    
    # Uncertainty penalty for rating calculation (rating = mu - k*sigma)
    uncertainty_penalty: float = 3.0
    
    # Optional race context overrides
    race_cluster: Optional[str] = None
    race_classification: Optional[str] = None
    race_date: Optional[str] = None
    race_distance: Optional[float] = None
    race_vertical_meters: Optional[float] = None


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


class ConfigBuilder:
    """
    Builds a full SimulationConfig from user parameters.
    
    Handles:
    - Validation of user inputs (team, race, scheme)
    - Resolution of internal data paths
    - Auto-population of race context from catalog
    - Time horizon calculations
    """
    
    def __init__(self, registry: Optional[DataRegistry] = None):
        """
        Initialize the config builder.
        
        Args:
            registry: Optional custom DataRegistry instance
        """
        self.registry = registry or get_registry()
    
    def build(self, config: UserConfig) -> SimulationConfig:
        """
        Build a full SimulationConfig from user parameters.
        
        Args:
            config: User configuration with parameters
            
        Returns:
            Complete SimulationConfig ready for simulation
            
        Raises:
            ConfigurationError: If configuration is invalid
            RaceNotFoundError: If race is not found
            TeamNotFoundError: If team is not found
            SchemeNotFoundError: If scheme is invalid
        """
        errors = []
        
        # Validate and normalize team
        try:
            team_name = self.registry.validate_team(config.team_name)
        except TeamNotFoundError as e:
            errors.append(str(e))
            team_name = config.team_name
        
        # Validate and normalize scheme
        try:
            scheme = self.registry.validate_scheme(config.scheme)
        except SchemeNotFoundError as e:
            errors.append(str(e))
            scheme = config.scheme
        
        # Validate race and get context
        try:
            race_context_data = self.registry.validate_race(config.race_name)
        except RaceNotFoundError as e:
            errors.append(str(e))
            race_context_data = {}
        
        # Raise collected errors
        if errors:
            raise ConfigurationError(
                "Configuration validation failed:\n\n" + "\n\n".join(errors)
            )
        
        # Resolve data paths
        try:
            data_paths = self.registry.resolve_paths(scheme)
        except DataNotFoundError as e:
            raise ConfigurationError(f"Data resolution failed: {e}")
        
        # Build race context
        race_context = self._build_race_context(config, race_context_data)
        
        # Determine year
        year = config.year or datetime.now().year
        
        # Build paths configuration
        paths = SimulationPaths(
            model_path=data_paths.model_path,
            rider_features_path=data_paths.rider_features_path,
            trueskill_leader_path=data_paths.trueskill_leader_path,
            trueskill_team_path=data_paths.trueskill_team_path,
            feature_columns_path=data_paths.feature_columns_path,
            clusters=self.registry.DEFAULT_CLUSTERS,
            leader_feature_columns=self.registry.DEFAULT_LEADER_FEATURE_COLUMNS,
            teammate_feature_columns=self.registry.DEFAULT_TEAMMATE_FEATURE_COLUMNS,
        )
        
        # Build run configuration
        run = SimulationRunConfig(
            team_name=team_name,
            race_name=config.race_name,
            year=year,
            level=config.level,
            scheme=scheme,
            riders_pool=config.riders_pool,
            roster_size=config.roster_size,
            top_k=config.top_k,
            output_dir=os.path.abspath(config.output_dir),
            race_context=race_context,
            time_gap=config.time_horizon,
            exclude_riders=config.exclude_riders,
            include_riders=config.include_riders,
            uncertainty_penalty=config.uncertainty_penalty,
        )
        
        return SimulationConfig(paths=paths, run=run)
    
    def _build_race_context(
        self,
        config: UserConfig,
        catalog_data: Dict[str, Any],
    ) -> RaceContext:
        """
        Build race context from user config and catalog data.
        
        User overrides take precedence over catalog values.
        """
        # Start with catalog defaults
        cluster = catalog_data.get("cluster", "Hills, uphill finish")
        classification = catalog_data.get("classification", "WT")
        typical_date = catalog_data.get("typical_date", "06-01")  # MM-DD format
        distance = catalog_data.get("distance")
        vertical_meters = catalog_data.get("verticalMeters")
        
        # Apply user overrides
        if config.race_cluster:
            cluster = config.race_cluster
        if config.race_classification:
            classification = config.race_classification
        if config.race_distance:
            distance = config.race_distance
        if config.race_vertical_meters:
            vertical_meters = config.race_vertical_meters
        
        # Determine race date
        if config.race_date:
            race_date = config.race_date
        else:
            # Build date from year and typical date
            year = config.year or datetime.now().year
            race_date = f"{year}-{typical_date}"
        
        # Apply time horizon if specified
        if config.time_horizon and not config.race_date:
            # Adjust the date by subtracting time_horizon days
            from datetime import timedelta
            base_date = datetime.strptime(race_date, "%Y-%m-%d")
            adjusted_date = base_date - timedelta(days=config.time_horizon)
            race_date = adjusted_date.strftime("%Y-%m-%d")
        
        return RaceContext(
            cluster=cluster,
            classification=classification,
            date=race_date,
            distance=distance,
            verticalMeters=vertical_meters,
        )
    
    def validate_inputs(self, config: UserConfig) -> List[str]:
        """
        Validate inputs and return list of warnings/errors without raising.
        
        Useful for pre-validation before building.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation error/warning messages (empty if valid)
        """
        issues = []
        
        # Check team
        try:
            self.registry.validate_team(config.team_name)
        except TeamNotFoundError as e:
            issues.append(f"Team: {e}")
        
        # Check race
        try:
            self.registry.validate_race(config.race_name)
        except RaceNotFoundError as e:
            issues.append(f"Race: {e}")
        
        # Check scheme
        try:
            self.registry.validate_scheme(config.scheme)
        except SchemeNotFoundError as e:
            issues.append(f"Scheme: {e}")
        
        # Check riders_pool
        if len(config.riders_pool) != 4:
            issues.append(f"riders_pool must have exactly 4 values, got {len(config.riders_pool)}")
        elif any(n < 0 for n in config.riders_pool):
            issues.append("riders_pool values must be non-negative")
        
        total_pool = sum(config.riders_pool)
        if config.roster_size < 1:
            issues.append("roster_size must be at least 1")
        if config.roster_size > total_pool:
            issues.append(f"roster_size ({config.roster_size}) cannot exceed total pool size ({total_pool})")
        if config.time_horizon is not None and config.time_horizon < 0:
            issues.append("time_horizon must be non-negative")
        
        return issues


def create_config(
    team: str,
    race: str,
    scheme: str = "time_lag",
    riders_pool: Optional[List[int]] = None,
    roster_size: int = 8,
    year: Optional[int] = None,
    time_horizon: Optional[int] = None,
    output_dir: str = "results/simulation_results/",
    exclude_riders: Optional[List[str]] = None,
    include_riders: Optional[List[str]] = None,
    uncertainty_penalty: float = 3.0,
    **kwargs,
) -> SimulationConfig:
    """
    Convenience function to create a full configuration from simple parameters.
    
    Args:
        team: Team name (e.g., "Israel - Premier Tech")
        race: Race name (e.g., "Giro d'Italia")
        scheme: Weighting scheme (time_lag, equal_weight, position)
        riders_pool: List of 4 integers for riders per category:
                    [race_cluster_leader, gc_leader, race_cluster_teammate, gc_teammate]
                    Default: [4, 4, 4, 4] = 16 riders total
        roster_size: Number of cyclists per roster
        year: Season year (defaults to current year)
        time_horizon: Days before race for feature cutoff
        output_dir: Output directory for results
        exclude_riders: List of rider names to exclude (e.g., riders who left the team)
        include_riders: List of rider names to include (e.g., new signings)
        uncertainty_penalty: Multiplier k for rating = mu - k*sigma (default: 3.0)
        **kwargs: Additional parameters passed to UserConfig
        
    Returns:
        Complete SimulationConfig ready for use with RosterOptimizer
        
    Example:
        >>> config = create_config(
        ...     team="Israel - Premier Tech",
        ...     race="Giro d'Italia",
        ...     scheme="time_lag",
        ...     riders_pool=[5, 4, 4, 3],  # 16 riders total
        ...     roster_size=8,
        ...     exclude_riders=["WOODS Michael", "FUGLSANG Jakob"],
        ...     include_riders=["NEW Signing"],
        ...     uncertainty_penalty=2.0,  # Less conservative
        ... )
        >>> optimizer = RosterOptimizer.from_config(config)
    """
    # Default riders_pool if not provided
    if riders_pool is None:
        riders_pool = [4, 4, 4, 4]
    
    user_config = UserConfig(
        team_name=team,
        race_name=race,
        scheme=scheme,
        riders_pool=riders_pool,
        roster_size=roster_size,
        year=year,
        time_horizon=time_horizon,
        output_dir=output_dir,
        exclude_riders=exclude_riders,
        include_riders=include_riders,
        uncertainty_penalty=uncertainty_penalty,
        **kwargs,
    )
    
    builder = ConfigBuilder()
    return builder.build(user_config)


def list_available_races() -> List[str]:
    """Get list of available races from the catalog."""
    return get_registry().get_available_races()


def list_available_teams() -> List[str]:
    """Get list of available teams from the catalog."""
    return get_registry().get_available_teams()


def list_available_schemes() -> List[str]:
    """Get list of available schemes (with models)."""
    return get_registry().get_available_schemes()


def get_race_info(race_name: str) -> Dict[str, Any]:
    """
    Get information about a specific race.
    
    Args:
        race_name: Name of the race
        
    Returns:
        Dictionary with race context (cluster, classification, etc.)
    """
    return get_registry().validate_race(race_name)


def check_system_status() -> Dict[str, Any]:
    """
    Check the status of the data system.
    
    Returns:
        Dictionary with availability status of all resources
    """
    return get_registry().check_data_availability()
