"""
Data Registry Module

Manages internal data paths and loading for the simulation package.
Provides automatic discovery and validation of internal data resources.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# Package data directory (relative to this file's location)
# This file is in utils/, so we need to go up one level to package root
_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_UTILS_DIR)
_DATA_DIR = os.path.join(_PACKAGE_DIR, "data")


@dataclass
class DataPaths:
    """Resolved paths to internal data resources."""
    model_path: str
    rider_features_path: str
    trueskill_leader_path: str
    trueskill_team_path: str
    feature_columns_path: str
    race_catalog_path: str
    team_catalog_path: str
    scheme: str  # The scheme these paths are resolved for


class DataRegistryError(Exception):
    """Base exception for data registry errors."""
    pass


class SchemeNotFoundError(DataRegistryError):
    """Raised when requested scheme is not available."""
    pass


class RaceNotFoundError(DataRegistryError):
    """Raised when requested race is not found in catalog."""
    pass


class TeamNotFoundError(DataRegistryError):
    """Raised when requested team is not found in catalog."""
    pass


class DataNotFoundError(DataRegistryError):
    """Raised when required data files are missing."""
    pass


class DataRegistry:
    """
    Registry for internal data resources.
    
    Discovers and validates available models, datasets, and configurations
    stored in the package's data directory.
    """
    
    # Valid scheme types
    VALID_SCHEMES = ["time_lag", "equal_weight", "position"]
    
    # Default configuration values
    DEFAULT_CLUSTERS = [
        "Flat",
        "Hills, flat finish",
        "Hills, uphill finish",
        "Mountains, flat finish",
        "Mountains, uphill finish",
        "Time Trial",
    ]
    
    DEFAULT_LEADER_FEATURE_COLUMNS = [
        "race_cluster_leader_mu",
        "race_cluster_leader_sigma",
        "gc_leader_mu",
        "gc_leader_sigma",
    ]
    
    DEFAULT_TEAMMATE_FEATURE_COLUMNS = [
        "race_cluster_teammate_mu",
        "race_cluster_teammate_sigma",
        "gc_teammate_mu",
        "gc_teammate_sigma",
    ]
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the data registry.
        
        Args:
            data_dir: Optional custom data directory path. 
                      Defaults to package's internal data directory.
        """
        self.data_dir = data_dir or _DATA_DIR
        self._race_catalog: Optional[Dict] = None
        self._team_catalog: Optional[List[str]] = None
        self._validate_data_directory()
    
    def _validate_data_directory(self) -> None:
        """Validate that the data directory exists and has required structure."""
        if not os.path.exists(self.data_dir):
            raise DataNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                f"Please ensure the internal data directory is properly set up."
            )
    
    # -------------------------------------------------------------------------
    # Path Resolution
    # -------------------------------------------------------------------------
    
    def get_model_path(self, scheme: str) -> str:
        """
        Get the path to the model file for the specified scheme.
        
        Args:
            scheme: Weighting scheme (time_lag, equal_weight, position)
            
        Returns:
            Absolute path to the model file
            
        Raises:
            SchemeNotFoundError: If scheme is invalid or model not found
        """
        self._validate_scheme(scheme)
        model_path = os.path.join(self.data_dir, "models", scheme, "model.json")
        
        if not os.path.exists(model_path):
            available = self.get_available_schemes()
            raise SchemeNotFoundError(
                f"Model for scheme '{scheme}' not found at: {model_path}\n"
                f"Available schemes: {available}"
            )
        
        return model_path
    
    def get_dataset_paths(self, scheme: str) -> Dict[str, str]:
        """
        Get paths to all dataset files for a specific scheme.
        
        - rider_features.csv: Common across all schemes
        - trueskill_leader.csv: Common across all schemes (leader ratings don't depend on scheme)
        - {scheme}_trueskill_team.csv: Scheme-specific (team/helper ratings vary by weighting method)
        
        Args:
            scheme: Weighting scheme (time_lag, equal_weight, position)
        
        Returns:
            Dictionary with keys: rider_features, trueskill_leader, trueskill_team
            
        Raises:
            DataNotFoundError: If any required dataset is missing
            SchemeNotFoundError: If scheme is invalid
        """
        self._validate_scheme(scheme)
        datasets_dir = os.path.join(self.data_dir, "datasets")
        
        # rider_features and trueskill_leader are common across schemes
        # trueskill_team is scheme-specific
        paths = {
            "rider_features": os.path.join(datasets_dir, "rider_features.csv"),
            "trueskill_leader": os.path.join(datasets_dir, "trueskill_leader.csv"),
            "trueskill_team": os.path.join(datasets_dir, f"{scheme}_trueskill_team.csv"),
        }
        
        missing = []
        
        for name, path in paths.items():
            if not os.path.exists(path):
                missing.append(name)
        
        if missing:
            raise DataNotFoundError(
                f"Missing required dataset files for scheme '{scheme}': {', '.join(missing)}\n"
                f"Expected location: {datasets_dir}\n"
                f"Required files:\n"
                f"  - rider_features.csv (common)\n"
                f"  - trueskill_leader.csv (common)\n"
                f"  - {scheme}_trueskill_team.csv (scheme-specific)"
            )
        
        return paths
    
    def get_feature_columns_path(self) -> Optional[str]:
        """
        Get path to feature columns configuration.
        
        Note: Feature columns are now optional since they can be extracted
        directly from the model. This method returns None if the file doesn't exist.
        
        Returns:
            Path to feature_columns.json, or None if not found
        """
        path = os.path.join(self.data_dir, "config", "feature_columns.json")
        
        if not os.path.exists(path):
            return None  # Optional - features will be extracted from model
        
        return path
    
    def get_race_catalog_path(self) -> str:
        """Get path to race catalog."""
        return os.path.join(self.data_dir, "races", "race_catalog.json")
    
    def get_team_catalog_path(self) -> str:
        """Get path to team catalog."""
        return os.path.join(self.data_dir, "teams", "team_catalog.json")
    
    def resolve_paths(self, scheme: str) -> DataPaths:
        """
        Resolve all data paths for a given scheme.
        
        Args:
            scheme: Weighting scheme
            
        Returns:
            DataPaths object with all resolved paths
        """
        datasets = self.get_dataset_paths(scheme)
        
        return DataPaths(
            model_path=self.get_model_path(scheme),
            rider_features_path=datasets["rider_features"],
            trueskill_leader_path=datasets["trueskill_leader"],
            trueskill_team_path=datasets["trueskill_team"],
            feature_columns_path=self.get_feature_columns_path(),
            race_catalog_path=self.get_race_catalog_path(),
            team_catalog_path=self.get_team_catalog_path(),
            scheme=scheme,
        )
    
    # -------------------------------------------------------------------------
    # Catalog Loading
    # -------------------------------------------------------------------------
    
    def load_race_catalog(self) -> Dict[str, Any]:
        """
        Load the race catalog containing valid races and their context.
        
        Returns:
            Dictionary mapping race names to their context (cluster, classification, etc.)
            
        Raises:
            DataNotFoundError: If race catalog is missing
        """
        if self._race_catalog is not None:
            return self._race_catalog
        
        catalog_path = self.get_race_catalog_path()
        
        if not os.path.exists(catalog_path):
            raise DataNotFoundError(
                f"Race catalog not found: {catalog_path}\n"
                f"Please create a race_catalog.json file with available races.\n"
                f"Expected format:\n"
                f'{{\n'
                f'  "Giro d\'Italia": {{\n'
                f'    "cluster": "Hills, uphill finish",\n'
                f'    "classification": "WT",\n'
                f'    "typical_date": "05-01",\n'
                f'    "distance": 180.0,\n'
                f'    "verticalMeters": 3000.0\n'
                f'  }},\n'
                f'  ...\n'
                f'}}'
            )
        
        with open(catalog_path, "r", encoding="utf-8") as f:
            self._race_catalog = json.load(f)
        
        return self._race_catalog
    
    def load_team_catalog(self) -> List[str]:
        """
        Load the team catalog containing valid team names.
        
        Returns:
            List of valid team names
            
        Raises:
            DataNotFoundError: If team catalog is missing
        """
        if self._team_catalog is not None:
            return self._team_catalog
        
        catalog_path = self.get_team_catalog_path()
        
        if not os.path.exists(catalog_path):
            raise DataNotFoundError(
                f"Team catalog not found: {catalog_path}\n"
                f"Please create a team_catalog.json file with available teams.\n"
                f"Expected format:\n"
                f'[\n'
                f'  "Israel - Premier Tech",\n'
                f'  "UAE Team Emirates",\n'
                f'  ...\n'
                f']'
            )
        
        with open(catalog_path, "r", encoding="utf-8") as f:
            self._team_catalog = json.load(f)
        
        return self._team_catalog
    
    def load_feature_columns(self) -> List[str]:
        """
        Load feature column names from configuration.
        
        Returns:
            List of feature column names
        """
        path = self.get_feature_columns_path()
        
        with open(path, "r", encoding="utf-8") as f:
            columns = json.load(f)
        
        if not isinstance(columns, list):
            raise DataNotFoundError(
                f"Feature columns file must contain a JSON list of column names."
            )
        
        return columns
    
    # -------------------------------------------------------------------------
    # Validation Methods
    # -------------------------------------------------------------------------
    
    def _validate_scheme(self, scheme: str) -> None:
        """Validate that scheme is a valid option."""
        if scheme not in self.VALID_SCHEMES:
            raise SchemeNotFoundError(
                f"Invalid scheme: '{scheme}'\n"
                f"Valid schemes are: {', '.join(self.VALID_SCHEMES)}"
            )
    
    def validate_race(self, race_name: str) -> Dict[str, Any]:
        """
        Validate that a race exists and return its context.
        
        Args:
            race_name: Name of the race to validate
            
        Returns:
            Race context dictionary
            
        Raises:
            RaceNotFoundError: If race is not in catalog
        """
        catalog = self.load_race_catalog()
        
        # Try exact match first
        if race_name in catalog:
            return catalog[race_name]
        
        # Try case-insensitive match
        race_name_lower = race_name.lower()
        for name, context in catalog.items():
            if name.lower() == race_name_lower:
                return context
        
        # Try partial match
        matches = [name for name in catalog.keys() if race_name_lower in name.lower()]
        
        if matches:
            suggestion = f"\nDid you mean one of these?\n  - " + "\n  - ".join(matches[:5])
        else:
            suggestion = f"\nAvailable races:\n  - " + "\n  - ".join(list(catalog.keys())[:10])
            if len(catalog) > 10:
                suggestion += f"\n  ... and {len(catalog) - 10} more"
        
        raise RaceNotFoundError(
            f"Race not found: '{race_name}'{suggestion}"
        )
    
    def validate_team(self, team_name: str) -> str:
        """
        Validate that a team exists and return the normalized name.
        
        Args:
            team_name: Name of the team to validate
            
        Returns:
            Normalized team name from catalog
            
        Raises:
            TeamNotFoundError: If team is not in catalog
        """
        catalog = self.load_team_catalog()
        
        # Try exact match first
        if team_name in catalog:
            return team_name
        
        # Try case-insensitive match
        team_name_lower = team_name.lower()
        for name in catalog:
            if name.lower() == team_name_lower:
                return name
        
        # Try partial match
        matches = [name for name in catalog if team_name_lower in name.lower()]
        
        if matches:
            suggestion = f"\nDid you mean one of these?\n  - " + "\n  - ".join(matches[:5])
        else:
            suggestion = f"\nAvailable teams:\n  - " + "\n  - ".join(catalog[:10])
            if len(catalog) > 10:
                suggestion += f"\n  ... and {len(catalog) - 10} more"
        
        raise TeamNotFoundError(
            f"Team not found: '{team_name}'{suggestion}"
        )
    
    def validate_scheme(self, scheme: str) -> str:
        """
        Validate scheme and return normalized value.
        
        Args:
            scheme: Scheme to validate
            
        Returns:
            Normalized scheme name
            
        Raises:
            SchemeNotFoundError: If scheme is invalid
        """
        self._validate_scheme(scheme)
        return scheme
    
    # -------------------------------------------------------------------------
    # Discovery Methods
    # -------------------------------------------------------------------------
    
    def get_available_schemes(self) -> List[str]:
        """
        Get list of schemes that have models available.
        
        Returns:
            List of available scheme names
        """
        models_dir = os.path.join(self.data_dir, "models")
        if not os.path.exists(models_dir):
            return []
        
        available = []
        for scheme in self.VALID_SCHEMES:
            model_path = os.path.join(models_dir, scheme, "model.json")
            if os.path.exists(model_path):
                available.append(scheme)
        
        return available
    
    def get_available_races(self) -> List[str]:
        """
        Get list of available race names.
        
        Returns:
            List of race names from catalog
        """
        try:
            catalog = self.load_race_catalog()
            return list(catalog.keys())
        except DataNotFoundError:
            return []
    
    def get_available_teams(self) -> List[str]:
        """
        Get list of available team names.
        
        Returns:
            List of team names from catalog
        """
        try:
            return self.load_team_catalog()
        except DataNotFoundError:
            return []
    
    def check_data_availability(self) -> Dict[str, Any]:
        """
        Check availability of all required data resources.
        
        Returns:
            Dictionary with status of each resource type
        """
        status = {
            "data_directory_exists": os.path.exists(self.data_dir),
            "available_schemes": self.get_available_schemes(),
            "datasets": {
                "common": {},
                "by_scheme": {},
            },
            "race_catalog": False,
            "team_catalog": False,
            "feature_columns": False,
        }
        
        datasets_dir = os.path.join(self.data_dir, "datasets")
        
        # Check common datasets (rider_features and trueskill_leader)
        rider_features_path = os.path.join(datasets_dir, "rider_features.csv")
        trueskill_leader_path = os.path.join(datasets_dir, "trueskill_leader.csv")
        status["datasets"]["common"]["rider_features.csv"] = os.path.exists(rider_features_path)
        status["datasets"]["common"]["trueskill_leader.csv"] = os.path.exists(trueskill_leader_path)
        
        # Check scheme-specific datasets (only trueskill_team is scheme-specific)
        for scheme in self.VALID_SCHEMES:
            filename = f"{scheme}_trueskill_team.csv"
            path = os.path.join(datasets_dir, filename)
            status["datasets"]["by_scheme"][scheme] = {
                filename: os.path.exists(path)
            }
        
        # Check catalogs
        status["race_catalog"] = os.path.exists(self.get_race_catalog_path())
        status["team_catalog"] = os.path.exists(self.get_team_catalog_path())
        
        # Check feature columns (optional - can be extracted from model)
        feature_columns_path = self.get_feature_columns_path()
        status["feature_columns"] = feature_columns_path is not None
        
        return status


# Module-level singleton for convenience
_default_registry: Optional[DataRegistry] = None


def get_registry() -> DataRegistry:
    """
    Get the default data registry instance.
    
    Returns:
        DataRegistry singleton instance
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = DataRegistry()
    return _default_registry


def reset_registry() -> None:
    """Reset the default registry (useful for testing)."""
    global _default_registry
    _default_registry = None
