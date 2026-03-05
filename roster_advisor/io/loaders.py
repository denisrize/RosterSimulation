from __future__ import annotations

import json
from typing import List, Optional

import pandas as pd

from roster_advisor.models.xgb_wrapper import load_xgb_model


def load_feature_columns(feature_columns: Optional[List[str]], feature_columns_path: Optional[str]) -> List[str]:
    if feature_columns:
        return feature_columns
    if feature_columns_path is None:
        raise ValueError("Feature columns must be provided via feature_columns or feature_columns_path.")
    with open(feature_columns_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("feature_columns_path must point to a JSON list of feature names.")
    return data


def load_dataframe(path: str, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if parse_dates:
        for col in parse_dates:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
    return df


def load_model(path: str):
    return load_xgb_model(path)


def validate_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {', '.join(missing)}")
