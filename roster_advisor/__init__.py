"""
Roster Advisor - Cycling Team Recommendation Engine

Provides roster optimization and recommendations for professional cycling teams
using trained ranking models and simulation-based analysis.

Simple Usage:
    >>> from roster_advisor import create_config, RosterOptimizer
    >>> config = create_config(
    ...     team="Israel - Premier Tech",
    ...     race="Giro d'Italia",
    ...     scheme="time_lag",
    ... )
    >>> optimizer = RosterOptimizer.from_config(config)
    >>> results = optimizer.simulate_best_rosters(...)

Or use the CLI:
    $ roster-advisor run --team "Israel - Premier Tech" --race "Giro d'Italia" --scheme time_lag
"""

from .engine.optimizer import RosterOptimizer
from .utils.config import (
    create_config,
    UserConfig,
    ConfigBuilder,
    list_available_races,
    list_available_teams,
    list_available_schemes,
    get_race_info,
    check_system_status,
)
from .utils.data_registry import (
    DataRegistry,
    get_registry,
    RaceNotFoundError,
    TeamNotFoundError,
    SchemeNotFoundError,
    DataNotFoundError,
)

__all__ = [
    # Core optimizer
    "RosterOptimizer",
    # Config creation
    "create_config",
    "UserConfig",
    "ConfigBuilder",
    # Discovery functions
    "list_available_races",
    "list_available_teams",
    "list_available_schemes",
    "get_race_info",
    "check_system_status",
    # Registry
    "DataRegistry",
    "get_registry",
    # Exceptions
    "RaceNotFoundError",
    "TeamNotFoundError",
    "SchemeNotFoundError",
    "DataNotFoundError",
]
