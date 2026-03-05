from __future__ import annotations

import pandas as pd


def load_reference_races(rider_features: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Load reference races for a given season that will be used to simulate future races.

    The function first selects all races from the requested `year`. If no races are
    available for that year, it falls back to using races from the previous year.
    """
    reference_year = year
    reference_races = rider_features[rider_features["year"] == reference_year].copy()
    if len(reference_races) == 0:
        reference_year = year - 1
        reference_races = rider_features[rider_features["year"] == reference_year].copy()
    return reference_races


def select_reference_race(reference_races: pd.DataFrame, race_name: str, cluster: str) -> pd.DataFrame:
    """
    Select reference race(s) by exact race name and cluster from the reference data.

    The function first looks for an exact match on `race_name` within the specified
    `cluster`. If no exact match is found, it falls back to a case-insensitive
    substring search on the race name within the same cluster.

    Note: this simple matching logic could be extended in the future to more
    sophisticated similarity-based retrieval (e.g. selecting the most similar
    historical race using k-means clustering or other distance-based methods).
    """
    matching = reference_races[
        (reference_races["cluster"] == cluster) &
        (reference_races["race"] == race_name)
    ].copy()

    if len(matching) == 0:
        matching = reference_races[
            (reference_races["cluster"] == cluster) &
            (reference_races["race"].str.contains(race_name, case=False, na=False))
        ].copy()

    if len(matching) == 0:
        raise ValueError(f"No reference races found for cluster={cluster} and race={race_name}")

    return matching
