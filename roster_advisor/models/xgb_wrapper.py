from __future__ import annotations

import os
import json
from typing import List, Optional, Tuple
import xgboost as xgb


# Store model path when loading so we can read feature names from file
_model_paths: dict = {}


def load_xgb_model(model_path: str) -> xgb.Booster:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = xgb.Booster()
    model.load_model(model_path)
    # Store path for later feature extraction
    _model_paths[id(model)] = model_path
    return model


def predict_scores(model: xgb.Booster, feature_matrix, feature_names):
    dmatrix = xgb.DMatrix(feature_matrix, feature_names=feature_names)
    return model.predict(dmatrix)


def get_model_feature_names(model: xgb.Booster) -> Optional[List[str]]:
    """
    Extract feature names from a trained XGBoost model.
    
    Tries multiple methods:
    1. Model's feature_names attribute
    2. Model's saved config
    3. Reading directly from the model JSON file (most reliable for saved models)
    
    Args:
        model: Trained XGBoost Booster
        
    Returns:
        List of feature names in the order the model expects, or None if not available
    """
    # Method 1: Try model.feature_names attribute
    try:
        feature_names = model.feature_names
        if feature_names:
            return list(feature_names)
    except AttributeError:
        pass
    
    # Method 2: Try model config
    try:
        config = json.loads(model.save_config())
        if 'learner' in config and 'feature_names' in config['learner']:
            return config['learner']['feature_names']
    except:
        pass
    
    # Method 3: Read directly from model JSON file
    model_path = _model_paths.get(id(model))
    if model_path:
        features = get_feature_names_from_model_file(model_path)
        if features:
            return features
    
    return None


def get_feature_names_from_model_file(model_path: str) -> Optional[List[str]]:
    """
    Extract feature names directly from an XGBoost model JSON file.
    
    This is the most reliable method for models saved with save_model().
    
    Args:
        model_path: Path to the model.json file
        
    Returns:
        List of feature names, or None if not found
    """
    if not model_path.endswith('.json'):
        return None
    
    try:
        with open(model_path, 'r', encoding='utf-8') as f:
            model_json = json.load(f)
        
        # XGBoost JSON format stores feature names in learner.feature_names
        if 'learner' in model_json:
            learner = model_json['learner']
            
            # Try feature_names directly
            if 'feature_names' in learner:
                return learner['feature_names']
            
            # Try in attributes
            if 'attributes' in learner and 'feature_names' in learner['attributes']:
                # Feature names might be stored as a JSON string
                fn = learner['attributes']['feature_names']
                if isinstance(fn, str):
                    return json.loads(fn)
                return fn
        
        # Alternative location in some XGBoost versions
        if 'feature_names' in model_json:
            return model_json['feature_names']
            
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    
    return None


def export_model_features_to_json(model_path: str, output_path: str) -> List[str]:
    """
    Load a model and export its feature names to a JSON file.
    
    Args:
        model_path: Path to the XGBoost model
        output_path: Path to save the feature names JSON
        
    Returns:
        List of feature names
    """
    # Try reading directly from file first (most reliable)
    feature_names = get_feature_names_from_model_file(model_path)
    
    # Fall back to loading model and extracting
    if feature_names is None:
        model = load_xgb_model(model_path)
        feature_names = get_model_feature_names(model)
    
    if feature_names is None or len(feature_names) == 0:
        raise ValueError(
            f"Could not extract feature names from model: {model_path}\n"
            "The model may have been trained without feature names.\n"
            "When training, ensure you pass feature_names to xgb.DMatrix()."
        )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(feature_names, f, indent=2)
    
    print(f"✓ Exported {len(feature_names)} feature names to: {output_path}")
    return feature_names
