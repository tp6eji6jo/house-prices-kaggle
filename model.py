from __future__ import annotations
from typing import Any, Dict
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

# Models List：add model here
MODEL_REGISTRY = {
    "lgbm": LGBMRegressor,
    "rf": RandomForestRegressor,
}

# Hyperparameter --params 
DEFAULTS: Dict[str, Dict[str, Any]] = {
    "lgbm": dict(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    ),
    "rf": dict(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    ),
}

def build_model(name: str, **overrides: Any):

    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")

    params = {**DEFAULTS.get(key, {}), **overrides}
    ModelCls = MODEL_REGISTRY[key]
    return ModelCls(**params)