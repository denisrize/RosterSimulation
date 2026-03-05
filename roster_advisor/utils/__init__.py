"""
Utility modules for Roster Advisor.

Contains configuration, data registry, and type definitions.
"""

from roster_advisor.utils.config import (
    UserConfig,
    ConfigBuilder,
    ConfigurationError,
    create_config,
    list_available_races,
    list_available_teams,
    list_available_schemes,
    get_race_info,
    check_system_status,
)
from roster_advisor.utils.data_registry import (
    DataRegistry,
    get_registry,
    RaceNotFoundError,
    TeamNotFoundError,
    SchemeNotFoundError,
    DataNotFoundError,
)
from roster_advisor.utils.types import (
    RaceContext,
    SimulationConfig,
    SimulationPaths,
    SimulationRunConfig,
)
from roster_advisor.utils.download import (
    download_data,
    check_data_status,
    print_data_status,
)

__all__ = [
    # Config
    "UserConfig",
    "ConfigBuilder",
    "ConfigurationError",
    "create_config",
    "list_available_races",
    "list_available_teams",
    "list_available_schemes",
    "get_race_info",
    "check_system_status",
    # Registry
    "DataRegistry",
    "get_registry",
    "RaceNotFoundError",
    "TeamNotFoundError",
    "SchemeNotFoundError",
    "DataNotFoundError",
    # Types
    "RaceContext",
    "SimulationConfig",
    "SimulationPaths",
    "SimulationRunConfig",
    # Download
    "download_data",
    "check_data_status",
    "print_data_status",
]
