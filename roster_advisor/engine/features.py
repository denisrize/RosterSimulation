from __future__ import annotations

import pandas as pd


def add_roster_aggregations(roster_df: pd.DataFrame, roster_riders, top_helpers=5) -> pd.DataFrame:
    """
    Add aggregated teammate features for each rider in a roster.
    """
    for idx, row in roster_df.iterrows():
        current_rider = row["rider"]
        teammates = [r for r in roster_riders if r != current_rider]
        teammate_data = roster_df[roster_df["rider"].isin(teammates)]
        if len(teammate_data) == 0:
            continue

        if "race_cluster_teammate_mu" in teammate_data.columns:
            teammates_sorted = teammate_data.sort_values("race_cluster_teammate_mu", ascending=False)
            for i in range(min(top_helpers, len(teammates_sorted))):
                helper_row = teammates_sorted.iloc[i]
                roster_df.at[idx, f"roster_helper_{i+1}_mu_race_cluster"] = helper_row.get("race_cluster_teammate_mu", 25.0)
                roster_df.at[idx, f"roster_helper_{i+1}_sigma_race_cluster"] = helper_row.get("race_cluster_teammate_sigma", 8.333)
            for i in range(len(teammates_sorted), top_helpers):
                roster_df.at[idx, f"roster_helper_{i+1}_mu_race_cluster"] = 25.0
                roster_df.at[idx, f"roster_helper_{i+1}_sigma_race_cluster"] = 8.333

            roster_df.at[idx, "roster_mean_mu_race_cluster"] = teammate_data["race_cluster_teammate_mu"].mean()
            roster_df.at[idx, "roster_mean_sigma_race_cluster"] = teammate_data["race_cluster_teammate_sigma"].mean()
            mean_sigma = teammate_data["race_cluster_teammate_sigma"].mean()
            roster_df.at[idx, "roster_mean_mu_sigma_ratio_race_cluster"] = (
                teammate_data["race_cluster_teammate_mu"].mean() / mean_sigma
                if mean_sigma > 0 else 1.0
            )

        if "gc_teammate_mu" in teammate_data.columns:
            teammates_sorted_gc = teammate_data.sort_values("gc_teammate_mu", ascending=False)
            for i in range(min(top_helpers, len(teammates_sorted_gc))):
                helper_row = teammates_sorted_gc.iloc[i]
                roster_df.at[idx, f"roster_helper_{i+1}_mu_gc"] = helper_row.get("gc_teammate_mu", 25.0)
                roster_df.at[idx, f"roster_helper_{i+1}_sigma_gc"] = helper_row.get("gc_teammate_sigma", 8.333)
            for i in range(len(teammates_sorted_gc), top_helpers):
                roster_df.at[idx, f"roster_helper_{i+1}_mu_gc"] = 25.0
                roster_df.at[idx, f"roster_helper_{i+1}_sigma_gc"] = 8.333

            roster_df.at[idx, "roster_mean_mu_gc"] = teammate_data["gc_teammate_mu"].mean()
            roster_df.at[idx, "roster_mean_sigma_gc"] = teammate_data["gc_teammate_sigma"].mean()
            mean_sigma_gc = teammate_data["gc_teammate_sigma"].mean()
            roster_df.at[idx, "roster_mean_mu_sigma_ratio_gc"] = (
                teammate_data["gc_teammate_mu"].mean() / mean_sigma_gc
                if mean_sigma_gc > 0 else 1.0
            )

        roster_df.at[idx, "roster_size"] = len(teammates) + 1

    return roster_df


def reconstruct_team_roster_features(race_df: pd.DataFrame, target_team: str, top_helpers=5) -> pd.DataFrame:
    """
    Reconstruct roster aggregation features for a single team in a race.
    """
    team_riders = race_df[race_df["team"] == target_team].copy()
    if len(team_riders) <= 1:
        return race_df

    for idx in team_riders.index:
        rider_name = race_df.at[idx, "rider"]
        teammates = team_riders[team_riders["rider"] != rider_name]
        if len(teammates) == 0:
            continue

        if "race_cluster_teammate_mu" in teammates.columns:
            teammates_sorted = teammates.sort_values("race_cluster_teammate_mu", ascending=False)
            for i in range(min(top_helpers, len(teammates_sorted))):
                helper_row = teammates_sorted.iloc[i]
                race_df.at[idx, f"roster_helper_{i+1}_mu_race_cluster"] = helper_row.get("race_cluster_teammate_mu", 25.0)
                race_df.at[idx, f"roster_helper_{i+1}_sigma_race_cluster"] = helper_row.get("race_cluster_teammate_sigma", 8.333)
            for i in range(len(teammates_sorted), top_helpers):
                race_df.at[idx, f"roster_helper_{i+1}_mu_race_cluster"] = 25.0
                race_df.at[idx, f"roster_helper_{i+1}_sigma_race_cluster"] = 8.333

            race_df.at[idx, "roster_mean_mu_race_cluster"] = teammates["race_cluster_teammate_mu"].mean()
            race_df.at[idx, "roster_mean_sigma_race_cluster"] = teammates["race_cluster_teammate_sigma"].mean()
            mean_sigma = teammates["race_cluster_teammate_sigma"].mean()
            race_df.at[idx, "roster_mean_mu_sigma_ratio_race_cluster"] = (
                teammates["race_cluster_teammate_mu"].mean() / mean_sigma
                if mean_sigma > 0 else 1.0
            )

        if "gc_teammate_mu" in teammates.columns:
            teammates_sorted_gc = teammates.sort_values("gc_teammate_mu", ascending=False)
            for i in range(min(top_helpers, len(teammates_sorted_gc))):
                helper_row = teammates_sorted_gc.iloc[i]
                race_df.at[idx, f"roster_helper_{i+1}_mu_gc"] = helper_row.get("gc_teammate_mu", 25.0)
                race_df.at[idx, f"roster_helper_{i+1}_sigma_gc"] = helper_row.get("gc_teammate_sigma", 8.333)
            for i in range(len(teammates_sorted_gc), top_helpers):
                race_df.at[idx, f"roster_helper_{i+1}_mu_gc"] = 25.0
                race_df.at[idx, f"roster_helper_{i+1}_sigma_gc"] = 8.333

            race_df.at[idx, "roster_mean_mu_gc"] = teammates["gc_teammate_mu"].mean()
            race_df.at[idx, "roster_mean_sigma_gc"] = teammates["gc_teammate_sigma"].mean()
            mean_sigma_gc = teammates["gc_teammate_sigma"].mean()
            race_df.at[idx, "roster_mean_mu_sigma_ratio_gc"] = (
                teammates["gc_teammate_mu"].mean() / mean_sigma_gc
                if mean_sigma_gc > 0 else 1.0
            )

        race_df.at[idx, "roster_size"] = len(teammates) + 1

    return race_df
