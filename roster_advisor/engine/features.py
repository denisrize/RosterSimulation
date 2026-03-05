from __future__ import annotations

import pandas as pd

def reconstruct_team_roster_features(race_df: pd.DataFrame, target_team: str, top_helpers=5) -> pd.DataFrame:
    """
    Reconstruct roster aggregation features for a single team in a race, following
    the same feature-engineering logic used in the VeloRost-Ex framework.

    For the given `target_team`, this function aggregates the strengths of all
    teammates around each rider in the team. For every rider, it:

    - Treats that rider as a potential leader in the evaluated roster combination.
    - Sorts teammates by their relevant skill estimates (e.g. race-cluster and GC
      teammate ratings) and assigns the top `top_helpers` as helper features.
    - Fills in default prior values when fewer than `top_helpers` teammates are
      available.
    - Computes team-level summary statistics (means and mean/sigma ratios) over
      the teammates' skills for both race-cluster and GC contexts.

    The result is a set of roster-level features that encode how strong the team
    around each potential leader is in the current evaluated combination of riders.
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
