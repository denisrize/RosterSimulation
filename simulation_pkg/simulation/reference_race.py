from __future__ import annotations

import pandas as pd


def load_reference_races(rider_features: pd.DataFrame, year: int) -> pd.DataFrame:
    reference_year = year
    reference_races = rider_features[rider_features["year"] == reference_year].copy()
    if len(reference_races) == 0:
        reference_year = year - 1
        reference_races = rider_features[rider_features["year"] == reference_year].copy()
    return reference_races


def select_reference_race(reference_races: pd.DataFrame, race_name: str, cluster: str) -> pd.DataFrame:
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
