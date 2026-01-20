from __future__ import annotations

import os
import xgboost as xgb


def load_xgb_model(model_path: str) -> xgb.Booster:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = xgb.Booster()
    model.load_model(model_path)
    return model


def predict_scores(model: xgb.Booster, feature_matrix, feature_names):
    dmatrix = xgb.DMatrix(feature_matrix, feature_names=feature_names)
    return model.predict(dmatrix)
