"""
Roster Optimizer Module

Simulates and optimizes team rosters by evaluating combinations of riders
for a given race using a trained model.
"""

from __future__ import annotations

import os
import json
from itertools import combinations
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

from ..io.loaders import load_dataframe, load_feature_columns, load_hyperparams, load_model
from ..models.xgb_wrapper import predict_scores
from .features import add_roster_aggregations, reconstruct_team_roster_features
from .reference_race import load_reference_races, select_reference_race


class RosterOptimizer:
    """
    Optimizes team rosters by simulating all possible rider combinations
    and predicting their performance using a trained model.
    """

    def __init__(
        self,
        model_path: str,
        rider_features_path: str,
        trueskill_leader_path: str,
        trueskill_team_path: str,
        feature_columns: list,
        hyperparams_path: Optional[str] = None,
        leader_feature_columns: Optional[list] = None,
        teammate_feature_columns: Optional[list] = None,
        clusters: Optional[list] = None,
        race_class: str = "all",
        scheme: str = "time_lag",
        year: int = 2026,
        level: str = "rider",
        time_gap: Optional[int] = None,
        exp_name: str = "class_features",
    ):
        self.model_path = model_path
        self.hyperparams_path = hyperparams_path
        self.rider_features_path = rider_features_path
        self.trueskill_leader_path = trueskill_leader_path
        self.trueskill_team_path = trueskill_team_path
        self.race_class = race_class
        self.scheme = scheme
        self.year = year
        self.level = level
        self.time_gap = time_gap
        self.exp_name = exp_name

        self.feature_columns = feature_columns
        self.leader_feature_columns = leader_feature_columns
        self.teammate_feature_columns = teammate_feature_columns
        self.clusters = clusters

        self.hyperparams = self._load_hyperparameters()
        self.model = load_model(self.model_path) if model_path else None
        if self.model is None and model_path:
            raise ValueError(f"Failed to load model from: {model_path}")

        self.rider_features = self._load_rider_features()
        self.trueskill_ratings = self._load_trueskill_ratings()

        if self.clusters is None:
            self.clusters = sorted(self.rider_features["cluster"].dropna().unique().tolist())

        print(f"✓ RosterOptimizer initialized")
        print(f"  Model: {model_path}")
        print(f"  Race class: {race_class}")
        print(f"  Scheme: {scheme}")
        print(f"  Year: {year}")
        print(f"  Rider features loaded: {len(self.rider_features)} records")
        print(f"  TrueSkill ratings loaded: {len(self.trueskill_ratings)} records")

    def _load_hyperparameters(self):
        if self.hyperparams_path and os.path.exists(self.hyperparams_path):
            try:
                hyperparams = load_hyperparams(self.hyperparams_path)
                print(f"✓ Hyperparameters loaded from: {self.hyperparams_path}")
                if hyperparams:
                    print("  Model parameters:")
                    for key, value in hyperparams.items():
                        if key != "optimal_num_boost_round":
                            print(f"    {key}: {value}")
                    if "optimal_num_boost_round" in hyperparams:
                        print(f"  Training rounds: {hyperparams['optimal_num_boost_round']}")
                return hyperparams
            except Exception as e:
                print(f"⚠ Could not load hyperparameters: {e}")
        else:
            print(f"⚠ No hyperparameters file provided")
        return None

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

    def get_team_rider_pool(self, team_name, race_context=None, N=18):
        if race_context is None:
            raise ValueError("Race context is required")
        cutoff_date = race_context["date"]

        print(f"\n{'='*80}")
        print(f"SELECTING RIDER POOL FOR: {team_name}")
        print(f"{'='*80}")
        print(f"Cutoff date: {cutoff_date}")
        print(f"Target pool size: {N} riders")

        rider_left = ["GEE Derek", "RICCITELLO Matthew", "WOODS Michael", "FUGLSANG Jakob", "CLARKE Simon"]
        print(f"✓ Excluding riders: {rider_left}")

        possible_riders = self.rider_features[
            (self.rider_features["team"] == team_name) &
            (self.rider_features["year"] == self.year) &
            (~self.rider_features["rider"].isin(rider_left))
        ]["rider"].unique()

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

        n_per_category = N // 3
        selected_riders = set()

        if "race_cluster_leader_mu" in latest_ratings.columns:
            top_leader = latest_ratings.nlargest(n_per_category, "race_cluster_leader_mu")
            selected_riders.update(top_leader["rider"].tolist())
            print(f"✓ Selected {len(top_leader)} riders by race_cluster_leader_mu")

        if "gc_leader_mu" in latest_ratings.columns:
            top_gc = latest_ratings.nlargest(n_per_category, "gc_leader_mu")
            selected_riders.update(top_gc["rider"].tolist())
            print(f"✓ Selected {len(selected_riders)} riders (added gc_leader_mu)")

        if "race_cluster_teammate_mu" in latest_ratings.columns:
            top_teammate = latest_ratings.nlargest(n_per_category, "race_cluster_teammate_mu")
            selected_riders.update(top_teammate["rider"].tolist())
            print(f"✓ Selected {len(selected_riders)} riders (added teammate_mu)")

        if len(selected_riders) < N and "gc_teammate_mu" in latest_ratings.columns:
            remaining_riders = latest_ratings[~latest_ratings["rider"].isin(selected_riders)]
            additional = remaining_riders.nlargest(N - len(selected_riders), "gc_teammate_mu")
            selected_riders.update(additional["rider"].tolist())
            print(f"✓ Added {len(additional)} additional riders to reach {N}")

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

        print(f"✓ Reference race loaded: {len(reference_race_complete)} competitors with features and ratings")

        simulated_race = reference_race_complete[reference_race_complete["team"] != team_name].copy()

        roster_data = []
        for rider in roster_riders:
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

        print(f"✓ Test roster added: {len(roster_df)} riders")
        print(f"✓ Total competitors: {len(simulated_race)} riders")

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

    def _update_top_combinations(self, new_result, top_combinations, max_unique=10):
        top_combinations.append(new_result)
        top_combinations.sort(key=lambda x: (x["best_rank"], -x["top_10_count"]))
        seen_performances = set()
        filtered_combinations = []

        for combo in top_combinations:
            perf_key = (combo["best_rank"], combo["top_10_count"])
            if len(seen_performances) < max_unique or perf_key in seen_performances:
                filtered_combinations.append(combo)
                seen_performances.add(perf_key)

        was_added = new_result in filtered_combinations
        return filtered_combinations, was_added

    def _should_consider_combination(self, new_result, top_combinations, max_unique=10):
        if len(top_combinations) == 0:
            return True
        seen_performances = {(combo["best_rank"], combo["top_10_count"]) for combo in top_combinations}
        if len(seen_performances) < max_unique:
            return True

        worst_combo = max(top_combinations, key=lambda x: (x["best_rank"], -x["top_10_count"]))
        new_perf = (new_result["best_rank"], new_result["top_10_count"])
        worst_perf = (worst_combo["best_rank"], worst_combo["top_10_count"])

        if new_perf[0] < worst_perf[0]:
            return True
        if new_perf[0] == worst_perf[0] and new_perf[1] >= worst_perf[1]:
            return True
        return False

    def _save_top_combinations_csv(self, top_combinations, save_path, team_name, race_name, race_cluster):
        if len(top_combinations) == 0:
            return

        save_data = []
        for combo in top_combinations:
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

        team_safe = team_name.replace(" ", "_").replace("–", "-")
        race_base_name = race_name.split(" | ")[0]
        filename = f"{save_path}/{team_safe}_{race_base_name}_{race_cluster}_top10_progress.csv"

        os.makedirs(save_path, exist_ok=True)
        df.to_csv(filename, index=False)

        metadata_file = f"{save_path}/{team_safe}_{race_base_name}_{race_cluster}_README.txt"
        with open(metadata_file, "w") as f:
            f.write("ROSTER SIMULATION RESULTS - COLUMN DESCRIPTIONS\n")
            f.write("=" * 60 + "\n\n")
            f.write("Base Columns:\n")
            f.write("  combo_id: Combination identifier number\n")
            f.write("  best_rank: Best finishing position achieved by any team rider\n")
            f.write("  best_rider: Name of the rider who achieved best_rank\n")
            f.write("  mean_rank: Average finishing position across roster\n")
            f.write("  top_5_count: Number of team riders finishing in top 5\n")
            f.write("  top_10_count: Number of team riders finishing in top 10\n")
            f.write("  top_30_count: Number of team riders finishing in top 30\n")
            f.write("  total_competitors: Total number of riders in the race\n\n")
            f.write("Rider Details (sorted by performance, best to worst):\n")
            f.write("  rider_N: Name of Nth best performing team rider\n")
            f.write("  race_rank_N: Predicted finishing position among ALL competitors\n")
            f.write("  score_N: Model prediction score for this rider\n\n")
            f.write("Note: Riders are sorted within each roster by their race_rank\n")

        return filename

    def simulate_best_rosters(
        self,
        team_name,
        race_name,
        race_context=None,
        N=18,
        n_riders_per_roster=8,
        save_path="results/roster_sims/",
        cutoff_date=None,
        top_k=10,
    ):
        print(f"\n{'='*80}")
        print(f"SIMULATING BEST ROSTERS")
        print(f"{'='*80}")
        print(f"Team: {team_name}")
        print(f"Race: {race_name}")
        print(f"Roster size: {n_riders_per_roster}")
        print(f"Search space: {N} riders")

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

        rider_pool = self.get_team_rider_pool(team_name, race_context=race_context, N=N)
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
        top_combinations = []
        save_interval = 1

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

                if self._should_consider_combination(performance, top_combinations, max_unique=10):
                    prev_best = top_combinations[0] if len(top_combinations) > 0 else None
                    top_combinations, was_added = self._update_top_combinations(
                        performance, top_combinations, max_unique=10
                    )
                    if was_added and len(top_combinations) > 0:
                        new_best = top_combinations[0]
                        if prev_best is None or (
                            new_best["best_rank"] < prev_best["best_rank"]
                            or (new_best["best_rank"] == prev_best["best_rank"]
                                and new_best["top_10_count"] > prev_best["top_10_count"])
                        ):
                            tqdm.write(
                                f"  🎯 New best! Rank {new_best['best_rank']} | Top-10: {new_best['top_10_count']} (combo #{i})"
                            )

                    if was_added and (i + 1) % save_interval == 0:
                        saved_file = self._save_top_combinations_csv(
                            top_combinations, save_path, team_name, race_name, reference_cluster
                        )
                        tqdm.write(f"  💾 Progress saved: {len(top_combinations)} top combinations to {saved_file}")

        if len(results) == 0:
            print("❌ No valid roster combinations could be evaluated")
            return None

        print(f"\n✓ Successfully evaluated {len(results):,} roster combinations")

        progress_file = self._save_top_combinations_csv(top_combinations, save_path, team_name, race_name, reference_cluster)
        print(f"✓ Top {len(top_combinations)} combinations saved to: {progress_file}")

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("best_rank", ascending=True)
        top_rosters = results_df.head(top_k).copy()

        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        team_safe = team_name.replace(" ", "_").replace("–", "-")
        race_safe = race_name.replace(" ", "_")

        output_file = f"{save_path}/{team_safe}_{race_safe}_{timestamp}_top{top_k}.csv"
        top_rosters.to_csv(output_file, index=False)

        all_results_file = f"{save_path}/{team_safe}_{race_safe}_{timestamp}_all.csv"
        results_df.to_csv(all_results_file, index=False)

        print(f"\n{'='*80}")
        print(f"SIMULATION COMPLETE")
        print(f"{'='*80}")
        print(f"✓ Top {top_k} rosters saved to: {output_file}")
        print(f"✓ All results saved to: {all_results_file}")

        return top_rosters
