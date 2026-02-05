"""
Roster Optimizer Module

Simulates and optimizes team rosters by evaluating combinations of riders
for a given race using a trained model.
"""

from __future__ import annotations

import os
import json
from itertools import combinations
from datetime import datetime, timedelta
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

from ..io.loaders import load_dataframe, load_feature_columns, load_model
from ..models.xgb_wrapper import predict_scores
from .features import add_roster_aggregations, reconstruct_team_roster_features
from .reference_race import load_reference_races, select_reference_race

if TYPE_CHECKING:
    from ..utils.types import SimulationConfig


class RosterOptimizer:
    """
    Optimizes team rosters by simulating all possible rider combinations
    and predicting their performance using a trained model.
    """

    @classmethod
    def from_config(cls, config: "SimulationConfig") -> "RosterOptimizer":
        """
        Create a RosterOptimizer from a SimulationConfig object.
        
        This is the recommended way to create an optimizer when using
        the simplified configuration API.
        
        Args:
            config: SimulationConfig object (from create_config or load_config)
            
        Returns:
            Configured RosterOptimizer instance
            
        Example:
            >>> from simulation_pkg import create_config, RosterOptimizer
            >>> config = create_config(
            ...     team="Israel - Premier Tech",
            ...     race="Giro d'Italia",
            ...     scheme="time_lag",
            ... )
            >>> optimizer = RosterOptimizer.from_config(config)
        """
        from ..io.loaders import load_feature_columns
        
        # Try to load feature columns, but they're optional since we can get them from the model
        feature_columns = None
        try:
            feature_columns = load_feature_columns(
                config.paths.feature_columns,
                config.paths.feature_columns_path,
            )
        except (ValueError, FileNotFoundError):
            # Feature columns will be extracted from the model instead
            pass
        
        return cls(
            model_path=config.paths.model_path,
            rider_features_path=config.paths.rider_features_path,
            trueskill_leader_path=config.paths.trueskill_leader_path,
            trueskill_team_path=config.paths.trueskill_team_path,
            feature_columns=feature_columns,
            leader_feature_columns=config.paths.leader_feature_columns,
            teammate_feature_columns=config.paths.teammate_feature_columns,
            clusters=config.paths.clusters,
            race_class="all",
            scheme=config.run.scheme,
            year=config.run.year,
            level=config.run.level,
            time_gap=config.run.time_gap,
        )

    def __init__(
        self,
        model_path: str,
        rider_features_path: str,
        trueskill_leader_path: str,
        trueskill_team_path: str,
        feature_columns: Optional[list] = None,
        leader_feature_columns: Optional[list] = None,
        teammate_feature_columns: Optional[list] = None,
        clusters: Optional[list] = None,
        race_class: str = "all",
        scheme: str = "time_lag",
        year: int = 2026,
        level: str = "rider",
        time_gap: Optional[int] = None,
    ):
        self.model_path = model_path
        self.rider_features_path = rider_features_path
        self.trueskill_leader_path = trueskill_leader_path
        self.trueskill_team_path = trueskill_team_path
        self.race_class = race_class
        self.scheme = scheme
        self.year = year
        self.level = level
        self.time_gap = time_gap # For future use (right now works only or start of the season)

        self.feature_columns = feature_columns
        self.leader_feature_columns = leader_feature_columns
        self.teammate_feature_columns = teammate_feature_columns
        self.clusters = clusters

        self.model = load_model(self.model_path) if model_path else None
        if self.model is None and model_path:
            raise ValueError(f"Failed to load model from: {model_path}")
        
        # Auto-extract and reorder feature columns from the model
        if self.model is not None:
            self.feature_columns = self._align_feature_columns_with_model()

        self.rider_features = self._load_rider_features()
        self.trueskill_ratings = self._load_trueskill_ratings()

        if self.clusters is None:
            self.clusters = sorted(self.rider_features["cluster"].dropna().unique().tolist())

        print(f"✓ RosterOptimizer initialized")
        print(f"  Model: {model_path}")
        print(f"  Race class: {race_class}")
        print(f"  Scheme: {scheme}")
        print(f"  Year: {year}")
        print(f"  Feature columns: {len(self.feature_columns)} (from model)")
        print(f"  Rider features loaded: {len(self.rider_features)} records")
        print(f"  TrueSkill ratings loaded: {len(self.trueskill_ratings)} records")

    def _align_feature_columns_with_model(self) -> list:
        """
        Extract feature column names from the model and use them as the authoritative source.
        
        This ensures features are always in the exact order the model was trained with,
        avoiding feature_names mismatch errors.
        
        Returns:
            List of feature column names in the order expected by the model
        """
        from ..models.xgb_wrapper import get_model_feature_names, get_feature_names_from_model_file
        
        # Try to get feature names from the loaded model
        model_features = get_model_feature_names(self.model)
        
        if model_features is not None and len(model_features) > 0:
            print(f"✓ Using feature order from model ({len(model_features)} features)")
            return model_features
        
        # Try reading directly from the model file
        if self.model_path:
            file_features = get_feature_names_from_model_file(self.model_path)
            if file_features is not None and len(file_features) > 0:
                print(f"✓ Using feature order from model file ({len(file_features)} features)")
                return file_features
        
        # Fallback to provided feature_columns if model doesn't have feature names
        if self.feature_columns and len(self.feature_columns) > 0:
            print("⚠ Model doesn't contain feature names, using provided feature_columns")
            print(f"  (If you get feature mismatch errors, ensure feature_columns.json matches model)")
            return self.feature_columns
        
        raise ValueError(
            "Could not determine feature columns.\n"
            "The model file doesn't contain feature names and no feature_columns.json was provided.\n"
            "Solutions:\n"
            "  1. Re-train the model with feature_names in DMatrix\n"
            "  2. Create data/config/feature_columns.json with the correct feature order\n"
            "  3. Use 'roster-sim model-features' to check if model has feature names"
        )

    def _load_rider_features(self):
        df = load_dataframe(self.rider_features_path, parse_dates=["date"])
        if "year" not in df.columns and "date" in df.columns:
            df["year"] = df["date"].dt.year
        print(f"✓ Rider features loaded: {len(df)} records")
        return df

    def _load_trueskill_ratings(self):
        leader_df = load_dataframe(self.trueskill_leader_path, parse_dates=["date"])
        team_df = load_dataframe(self.trueskill_team_path, parse_dates=["date"])

        ratings_df = pd.merge(
            leader_df,
            team_df,
            on=["rider", "date", "race", "team", "cluster", "classification"],
            how="inner",
            suffixes=("_leader", "_team"),
        )

        if self.leader_feature_columns is None:
            self.leader_feature_columns = [
                col for col in leader_df.columns
                if col not in {"rider", "date", "race", "team", "cluster", "classification"}
            ]
        if self.teammate_feature_columns is None:
            self.teammate_feature_columns = [
                col for col in team_df.columns
                if col not in {"rider", "date", "race", "team", "cluster", "classification"}
            ]

        print(f"✓ TrueSkill ratings loaded: {len(ratings_df)} records")
        return ratings_df

    def extract_race_features(self, race_result):
        race_result = race_result.copy()

        def race_class_to_ordinal(race_class):
            if race_class == "WT":
                return 4
            if race_class == "Pro":
                return 3
            if race_class == "1":
                return 2
            return 1

        if "race_class" in race_result.columns:
            race_result["race_class_ord"] = race_result["race_class"].apply(race_class_to_ordinal)
        race_result["StageRace"] = race_result["race"].apply(lambda x: 1 if "Stage" in x else 0)
        parcours = pd.get_dummies(race_result["cluster"], prefix="cluster")
        race_result = pd.concat([race_result, parcours], axis=1)
        parcours_cols = list(parcours.columns)
        race_result[parcours_cols] = race_result[parcours_cols].astype(int)
        return race_result

    def get_team_rider_pool(
        self,
        team_name,
        race_context=None,
        riders_pool: Optional[list] = None,
        exclude_riders: Optional[list] = None,
        include_riders: Optional[list] = None,
        uncertainty_penalty: float = 3.0,
    ):
        """
        Select the rider pool for simulation.
        
        Args:
            team_name: Team to select riders from
            race_context: Race context with date, cluster, etc.
            riders_pool: List of 4 integers specifying how many NEW unique riders to select
                        from each category: [race_cluster_leader, gc_leader, 
                        race_cluster_teammate, gc_teammate]. Default: [4, 4, 4, 4]
            exclude_riders: List of rider names to exclude (e.g., riders who left the team)
            include_riders: List of rider names to include even if not on team roster
                           (e.g., new signings not yet in the data)
            uncertainty_penalty: Multiplier for sigma in rating calculation.
                                rating = mu - k * sigma (default: 3.0)
                                Higher values penalize uncertain ratings more.
        """
        if race_context is None:
            raise ValueError("Race context is required")
        
        # Default riders_pool if not provided
        if riders_pool is None:
            riders_pool = [4, 4, 4, 4]
        
        if len(riders_pool) != 4:
            raise ValueError(f"riders_pool must have exactly 4 values, got {len(riders_pool)}")
        
        total_pool_size = sum(riders_pool)
        cutoff_date = race_context["date"] + pd.DateOffset(years=1)

        print(f"\n{'='*80}")
        print(f"SELECTING RIDER POOL FOR: {team_name}")
        print(f"{'='*80}")
        print(f"Cutoff date: {cutoff_date}")
        print(f"Riders pool distribution: {riders_pool}")
        print(f"  - race_cluster_leader: {riders_pool[0]}")
        print(f"  - gc_leader: {riders_pool[1]}")
        print(f"  - race_cluster_teammate: {riders_pool[2]}")
        print(f"  - gc_teammate: {riders_pool[3]}")
        print(f"Target pool size: {total_pool_size} riders")

        # Handle excluded riders (riders who left the team)
        exclude_riders = exclude_riders or []
        if exclude_riders:
            print(f"✓ Excluding riders: {exclude_riders}")

        # Get riders from team roster for the year, excluding specified riders
        possible_riders = self.rider_features[
            (self.rider_features["team"] == team_name) &
            (self.rider_features["year"] == self.year) &
            (~self.rider_features["rider"].isin(exclude_riders))
        ]["rider"].unique().tolist()

        # Handle included riders (new signings or riders to add)
        include_riders = include_riders or []
        if include_riders:
            # Add included riders if they exist in the data (from any team)
            for rider in include_riders:
                if rider not in possible_riders:
                    # Check if rider exists in the dataset
                    if rider in self.rider_features["rider"].values:
                        possible_riders.append(rider)
                        print(f"✓ Including rider: {rider}")
                    else:
                        print(f"⚠ Cannot include rider '{rider}' - not found in dataset")

        print(f"✓ {len(possible_riders)} possible riders found for {team_name} in {self.year}")

        team_ratings = self.trueskill_ratings[
            (self.trueskill_ratings["team"] == team_name) &
            (self.trueskill_ratings["date"] <= cutoff_date) &
            (self.trueskill_ratings["rider"].isin(possible_riders)) &
            (self.trueskill_ratings["cluster"] == race_context["cluster"])
        ].copy()

        if len(team_ratings) == 0:
            raise ValueError(f"No ratings found for team: {team_name}")

        print(f"✓ Found {len(team_ratings)} rating records for {team_name}")

        team_ratings = team_ratings.sort_values("date", ascending=False)
        latest_ratings = team_ratings.groupby("rider").first().reset_index()

        print(f"✓ {len(latest_ratings)} unique riders found")
        print(f"Using uncertainty penalty k={uncertainty_penalty} (rating = mu - k*sigma)")

        # Categories: (mu_column, sigma_column, target_count)
        categories = [
            ("race_cluster_leader_mu", "race_cluster_leader_sigma", riders_pool[0]),
            ("gc_leader_mu", "gc_leader_sigma", riders_pool[1]),
            ("race_cluster_teammate_mu", "race_cluster_teammate_sigma", riders_pool[2]),
            ("gc_teammate_mu", "gc_teammate_sigma", riders_pool[3]),
        ]
        
        selected_riders = set()
        
        for mu_col, sigma_col, target_count in categories:
            if target_count == 0:
                continue
                
            if mu_col not in latest_ratings.columns:
                print(f"⚠ Column '{mu_col}' not found, skipping")
                continue
            
            # Calculate conservative rating = mu - k * sigma
            if sigma_col in latest_ratings.columns:
                latest_ratings[f"_rating_{mu_col}"] = (
                    latest_ratings[mu_col] - uncertainty_penalty * latest_ratings[sigma_col]
                )
                sort_col = f"_rating_{mu_col}"
                rating_info = f"(mu - {uncertainty_penalty}*sigma)"
            else:
                # Fallback to mu only if sigma not available
                sort_col = mu_col
                rating_info = "(mu only, sigma not found)"
                print(f"⚠ Column '{sigma_col}' not found, using mu only")
            
            # Select riders until we've added target_count NEW unique riders
            added_count = 0
            sorted_by_rating = latest_ratings.sort_values(sort_col, ascending=False)
            
            for _, row in sorted_by_rating.iterrows():
                rider = row["rider"]
                if rider not in selected_riders:
                    selected_riders.add(rider)
                    added_count += 1
                    if added_count >= target_count:
                        break
            
            print(f"✓ Added {added_count} riders from {mu_col} {rating_info} (pool size: {len(selected_riders)})")

        rider_pool = latest_ratings[latest_ratings["rider"].isin(selected_riders)].copy()
        print(f"\n✓ Final rider pool: {len(rider_pool)} riders")
        print(f"  Riders: {', '.join(rider_pool['rider'].tolist()[:5])}...")
        return rider_pool

    def generate_roster_combinations(self, rider_pool, n_riders_per_roster=8):
        riders = rider_pool["rider"].tolist()
        n_riders = len(riders)

        print(f"\n{'='*80}")
        print(f"GENERATING ROSTER COMBINATIONS")
        print(f"{'='*80}")
        print(f"Pool size: {n_riders} riders")
        print(f"Roster size: {n_riders_per_roster} riders")

        if n_riders < n_riders_per_roster:
            raise ValueError(f"Not enough riders ({n_riders}) to form roster of {n_riders_per_roster}")

        from math import comb
        n_combinations = comb(n_riders, n_riders_per_roster)
        print(f"Total combinations: {n_combinations:,}")

        if n_combinations > 200000:
            print(f"⚠ Warning: {n_combinations:,} combinations may take a while to evaluate")

        roster_combos = list(combinations(riders, n_riders_per_roster))
        print(f"✓ Generated {len(roster_combos):,} roster combinations")
        return roster_combos

    def construct_roster_features(self, roster_riders, race_context):
        race_date = race_context.get("date")
        race_cluster = race_context.get("cluster")
        race_class_val = race_context.get("classification")

        roster_features = []
        for rider in roster_riders:
            rider_data = self.rider_features[
                (self.rider_features["rider"] == rider) &
                (self.rider_features["date"] <= race_date)
            ]
            if len(rider_data) == 0:
                rider_data = self.rider_features[self.rider_features["rider"] == rider]
            if len(rider_data) == 0:
                continue

            rider_record = rider_data.sort_values("date", ascending=False).iloc[0].copy()

            rider_ratings = self.trueskill_ratings[
                (self.trueskill_ratings["rider"] == rider) &
                (self.trueskill_ratings["date"] <= race_date)
            ]
            if len(rider_ratings) > 0:
                latest_rating = rider_ratings.sort_values("date", ascending=False).iloc[0]
                for col in (self.leader_feature_columns or []) + (self.teammate_feature_columns or []):
                    if col in latest_rating:
                        rider_record[col] = latest_rating[col]

            roster_features.append(rider_record)

        if len(roster_features) == 0:
            return None

        roster_df = pd.DataFrame(roster_features)
        roster_df = add_roster_aggregations(roster_df, roster_riders)

        roster_df["cluster"] = race_cluster
        roster_df["classification"] = race_class_val
        roster_df["date"] = race_date

        for cluster in self.clusters or []:
            roster_df[f"cluster_{cluster}"] = (roster_df["cluster"] == cluster).astype(int)

        return roster_df

    def predict_roster_performance(self, roster_riders, reference_race_name, reference_race_date, reference_cluster, team_name):
        reference_race_features = self.rider_features[
            (self.rider_features["race"] == reference_race_name) &
            (self.rider_features["date"] == reference_race_date) &
            (self.rider_features["cluster"] == reference_cluster)
        ].copy()

        if len(reference_race_features) == 0:
            print(f"⚠ No riders found for reference race: {reference_race_name}")
            return None

        reference_race_ratings = self.trueskill_ratings[
            (self.trueskill_ratings["race"] == reference_race_name) &
            (self.trueskill_ratings["date"] == reference_race_date) &
            (self.trueskill_ratings["cluster"] == reference_cluster)
        ].copy()

        reference_race_complete = pd.merge(
            reference_race_features,
            reference_race_ratings,
            on=["rider", "date", "race", "team", "cluster", "classification"],
            how="left",
            suffixes=("", "_rating"),
        )

        # print(f"✓ Reference race loaded: {len(reference_race_complete)} competitors with features and ratings")

        simulated_race = reference_race_complete[reference_race_complete["team"] != team_name].copy()

        roster_data = []
        for rider in roster_riders:
            # Use the most up to date skills and features
            rider_features = self.rider_features[
                (self.rider_features["rider"] == rider) &
                (self.rider_features["cluster"] == reference_cluster)
            ].sort_values("date", ascending=False)
            if len(rider_features) == 0:
                print(f"  ⚠ No features found for {rider} in cluster {reference_cluster}")
                continue

            rider_record = rider_features.iloc[0].copy()
            rider_ratings = self.trueskill_ratings[
                (self.trueskill_ratings["rider"] == rider) &
                (self.trueskill_ratings["cluster"] == reference_cluster)
            ].sort_values("date", ascending=False)

            if len(rider_ratings) > 0:
                latest_rating = rider_ratings.iloc[0]
                for col in (self.leader_feature_columns or []) + (self.teammate_feature_columns or []):
                    if col in latest_rating:
                        rider_record[col] = latest_rating[col]

            rider_record["team"] = team_name
            rider_record["race"] = reference_race_name
            rider_record["date"] = reference_race_date

            roster_data.append(rider_record)

        if len(roster_data) == 0:
            print(f"  ❌ No valid riders in test roster")
            return None

        roster_df = pd.DataFrame(roster_data)
        simulated_race = pd.concat([simulated_race, roster_df], ignore_index=True)

        # print(f"✓ Test roster added: {len(roster_df)} riders")
        # print(f"✓ Total competitors: {len(simulated_race)} riders")

        simulated_race = reconstruct_team_roster_features(simulated_race, team_name)
        simulated_race = self.extract_race_features(simulated_race)

        missing_features = [col for col in self.feature_columns if col not in simulated_race.columns]
        for col in missing_features:
            simulated_race[col] = 0

        try:
            X = simulated_race[self.feature_columns].values
            pred_scores = predict_scores(self.model, X, self.feature_columns)
            simulated_race["pred_score"] = pred_scores
        except Exception as e:
            print(f"⚠ Prediction error: {e}")
            return None

        simulated_race = simulated_race.sort_values("pred_score", ascending=False).reset_index(drop=True)
        simulated_race["predicted_rank"] = range(1, len(simulated_race) + 1)

        team_riders = simulated_race[simulated_race["team"] == team_name].copy()
        if len(team_riders) == 0:
            return None

        best_rank = team_riders["predicted_rank"].min()
        best_rider = team_riders.loc[team_riders["predicted_rank"].idxmin(), "rider"]
        mean_rank = team_riders["predicted_rank"].mean()
        top_5_count = (team_riders["predicted_rank"] <= 5).sum()
        top_10_count = (team_riders["predicted_rank"] <= 10).sum()
        top_30_count = (team_riders["predicted_rank"] <= 30).sum()

        rider_results = []
        for _, rider_row in team_riders.iterrows():
            rider_results.append({
                "rider": rider_row["rider"],
                "predicted_rank": int(rider_row["predicted_rank"]),
                "predicted_score": float(rider_row["pred_score"]),
            })

        return {
            "riders": roster_riders,
            "best_rider": best_rider,
            "best_rank": int(best_rank),
            "mean_rank": float(mean_rank),
            "top_5_count": int(top_5_count),
            "top_10_count": int(top_10_count),
            "top_30_count": int(top_30_count),
            "rider_results": rider_results,
            "total_competitors": len(simulated_race),
        }

    def _sanitize_filename(self, name: str) -> str:
        """
        Sanitize a string for use in filenames.
        Removes/replaces Windows-invalid characters: \\ / : * ? " < > |
        """
        sanitized = name.replace(" ", "_").replace("|", "-").replace(":", "-")
        sanitized = sanitized.replace("\\", "-").replace("/", "-").replace("*", "")
        sanitized = sanitized.replace("?", "").replace('"', "").replace("<", "").replace(">", "")
        sanitized = sanitized.replace("–", "-")  # en-dash
        return sanitized

    def _save_combinations_csv(
        self, 
        combinations, 
        save_path, 
        team_name, 
        race_name, 
        race_cluster,
        last_saved_index: int = 0,
        custom_filename: Optional[str] = None,
        sort_by: Optional[str] = None
    ):
        """
        Save combinations to CSV file.
        
        Args:
            combinations: List of all combinations evaluated so far
            save_path: Directory to save to
            team_name: Team name for filename
            race_name: Race name for filename
            race_cluster: Race cluster for filename
            last_saved_index: Index of last saved combination (append from here)
            custom_filename: If provided, use this filename and save all (no append)
            sort_by: If provided, sort DataFrame by this column before saving
            
        Returns:
            Tuple of (filename, new_last_saved_index)
        """
        if len(combinations) == 0:
            return None, 0

        # If custom filename provided, save ALL combinations (no incremental append)
        if custom_filename:
            combos_to_save = combinations
            append_mode = False
        else:
            # Only process new combinations since last save
            combos_to_save = combinations[last_saved_index:]
            append_mode = True
            
        if len(combos_to_save) == 0:
            team_safe = self._sanitize_filename(team_name)
            race_base_name = self._sanitize_filename(race_name.split(" | ")[0])
            filename = f"{save_path}/{team_safe}_{race_base_name}_{race_cluster}_progress.csv"
            return filename, last_saved_index

        save_data = []
        for combo in combos_to_save:
            row = {
                "combo_id": combo["combo_id"],
                "best_rank": combo["best_rank"],
                "best_rider": combo["best_rider"],
                "mean_rank": combo["mean_rank"],
                "top_5_count": combo["top_5_count"],
                "top_10_count": combo["top_10_count"],
                "top_30_count": combo["top_30_count"],
                "total_competitors": combo["total_competitors"],
            }
            if "rider_results" in combo:
                sorted_riders = sorted(combo["rider_results"], key=lambda x: x["predicted_rank"])
                for i, rider_result in enumerate(sorted_riders, 1):
                    row[f"rider_{i}"] = rider_result["rider"]
                    row[f"race_rank_{i}"] = rider_result["predicted_rank"]
                    row[f"score_{i}"] = round(rider_result["predicted_score"], 4)
            save_data.append(row)

        df = pd.DataFrame(save_data)
        base_cols = ["combo_id", "best_rank", "best_rider", "mean_rank",
                     "top_5_count", "top_10_count", "top_30_count", "total_competitors"]
        rider_cols = [col for col in df.columns if col not in base_cols]

        def sort_key(col):
            parts = col.split("_")
            if col.startswith("rider_"):
                return (int(parts[-1]), 0)
            if col.startswith("race_rank_"):
                return (int(parts[-1]), 1)
            if col.startswith("score_"):
                return (int(parts[-1]), 2)
            return (999, 999)

        rider_cols_sorted = sorted(rider_cols, key=sort_key)
        df = df[base_cols + rider_cols_sorted]
        
        # Sort if requested
        if sort_by and sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=True)

        # Determine filename
        if custom_filename:
            filename = custom_filename
        else:
            team_safe = self._sanitize_filename(team_name)
            race_base_name = self._sanitize_filename(race_name.split(" | ")[0])
            filename = f"{save_path}/{team_safe}_{race_base_name}_{race_cluster}_progress.csv"

        os.makedirs(save_path, exist_ok=True)
        
        # Custom filename: always overwrite; otherwise: append mode
        if custom_filename:
            df.to_csv(filename, index=False)
        else:
            file_exists = os.path.exists(filename) and last_saved_index > 0
            df.to_csv(filename, mode='a' if file_exists else 'w', header=not file_exists, index=False)
        
        return filename, len(combinations)

    def simulate_best_rosters(
        self,
        team_name,
        race_name,
        race_context=None,
        riders_pool: Optional[list] = None,
        n_riders_per_roster=8,
        save_path="results/simulation_results/",
        exclude_riders: Optional[list] = None,
        include_riders: Optional[list] = None,
        uncertainty_penalty: float = 3.0,
    ):
        """
        Simulate all roster combinations and save results.
        
        Args:
            team_name: Team to optimize
            race_name: Target race
            race_context: Race context (cluster, classification, date, etc.)
            riders_pool: List of 4 integers specifying riders per category:
                        [race_cluster_leader, gc_leader, race_cluster_teammate, gc_teammate]
                        Default: [4, 4, 4, 4] = 16 riders total
            n_riders_per_roster: Number of riders per roster
            save_path: Directory to save results
            exclude_riders: List of rider names to exclude (e.g., riders who left)
            include_riders: List of rider names to include (e.g., new signings)
            uncertainty_penalty: Multiplier k for rating = mu - k*sigma (default: 3.0)
            
        Returns:
            DataFrame with top results sorted by best_rank
        """
        # Default riders_pool if not provided
        if riders_pool is None:
            riders_pool = [4, 4, 4, 4]
            
        total_pool_size = sum(riders_pool)
        
        print(f"\n{'='*80}")
        print(f"SIMULATING BEST ROSTERS")
        print(f"{'='*80}")
        print(f"Team: {team_name}")
        print(f"Race: {race_name}")
        print(f"Roster size: {n_riders_per_roster}")
        print(f"Search space: {total_pool_size} riders (pool: {riders_pool})")
        print(f"Uncertainty penalty: k={uncertainty_penalty}")
        if exclude_riders:
            print(f"Excluding: {exclude_riders}")
        if include_riders:
            print(f"Including: {include_riders}")

        if self.model is None:
            raise ValueError("No model loaded. Provide model_path during initialization.")

        if race_context is None:
            race_context = {
                "race": race_name,
                "cluster": "Hills, uphill finish",
                "classification": "WT",
                "date": pd.Timestamp(f"{self.year}-12-31"),
                "distance": 180.0,
                "verticalMeters": 3000.0,
            }
            print(f"⚠ Using default race context: {race_context['cluster']}, {race_context['classification']}")

        if isinstance(race_context["date"], str):
            race_context["date"] = pd.to_datetime(race_context["date"])

        rider_pool = self.get_team_rider_pool(
            team_name,
            race_context=race_context,
            riders_pool=riders_pool,
            exclude_riders=exclude_riders,
            include_riders=include_riders,
            uncertainty_penalty=uncertainty_penalty,
        )
        roster_combos = self.generate_roster_combinations(rider_pool, n_riders_per_roster)

        print(f"\n{'='*80}")
        print(f"LOADING REFERENCE RACE DATA")
        print(f"{'='*80}")
        reference_races = load_reference_races(self.rider_features, self.year)

        cluster = race_context.get("cluster")
        matching_races = select_reference_race(reference_races, race_name, cluster)

        reference_race_name = matching_races.iloc[0]["race"]
        reference_race_date = matching_races.iloc[0]["date"]
        reference_cluster = matching_races.iloc[0]["cluster"]

        print(f"✓ Using reference race: {reference_race_name}")
        print(f"  Date: {reference_race_date}")
        print(f"  Cluster: {reference_cluster}")
        print(f"  Available records: {len(matching_races)}")

        print(f"\n{'='*80}")
        print(f"EVALUATING ROSTER COMBINATIONS")
        print(f"{'='*80}")

        results = []
        save_interval = 100  # Save every 100 combinations
        best_so_far = None
        last_saved_index = 0  # Track what's already been saved

        for i, roster_riders in enumerate(tqdm(roster_combos, desc="Evaluating rosters")):
            performance = self.predict_roster_performance(
                list(roster_riders),
                reference_race_name,
                reference_race_date,
                reference_cluster,
                team_name,
            )
            if performance is not None:
                performance["combo_id"] = i
                performance["team"] = team_name
                performance["race"] = race_name
                results.append(performance)

                # Track and report new best performance
                if best_so_far is None or (
                    performance["best_rank"] < best_so_far["best_rank"]
                    or (performance["best_rank"] == best_so_far["best_rank"]
                        and performance["top_10_count"] > best_so_far["top_10_count"])
                ):
                    best_so_far = performance
                    tqdm.write(
                        f"  🎯 New best! Rank {performance['best_rank']} | Top-10: {performance['top_10_count']} (combo #{i})"
                    )

                # Save progress periodically (append only new rows)
                if (i + 1) % save_interval == 0:
                    saved_file, last_saved_index = self._save_combinations_csv(
                        results, save_path, team_name, race_name, reference_cluster, last_saved_index
                    )
                    tqdm.write(f"  💾 Progress saved: {len(results)} combinations (+{save_interval} new) to {saved_file}")

        if len(results) == 0:
            print("❌ No valid roster combinations could be evaluated")
            return None

        print(f"\n✓ Successfully evaluated {len(results):,} roster combinations")

        # Final save - append any remaining combinations not yet saved
        remaining = len(results) - last_saved_index
        progress_file, _ = self._save_combinations_csv(
            results, save_path, team_name, race_name, reference_cluster, last_saved_index
        )
        if remaining > 0:
            print(f"✓ Final {remaining} combinations appended to: {progress_file}")
        print(f"✓ All {len(results)} combinations saved to: {progress_file}")

        # Save final sorted results (reuse the same save method with custom filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        team_safe = self._sanitize_filename(team_name)
        race_safe = self._sanitize_filename(race_name)
        all_results_file = f"{save_path}/{team_safe}_{race_safe}_{timestamp}_sorted.csv"
        
        self._save_combinations_csv(
            results, save_path, team_name, race_name, reference_cluster,
            custom_filename=all_results_file,
            sort_by="best_rank"
        )

        # Delete the progress file now that we have the final sorted version
        if progress_file and os.path.exists(progress_file):
            try:
                os.remove(progress_file)
                print(f"✓ Progress file deleted: {progress_file}")
            except OSError as e:
                print(f"⚠ Could not delete progress file: {e}")

        print(f"\n{'='*80}")
        print(f"SIMULATION COMPLETE")
        print(f"{'='*80}")
        print(f"✓ Final results saved to: {all_results_file}")

        # Return the sorted results as DataFrame
        return pd.read_csv(all_results_file)