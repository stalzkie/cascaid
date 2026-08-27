"""Flattened XGBoost baseline (PRD 6.2): same per-node features, no adjacency at all."""

from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier


def train_baseline(x_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="aucpr",
        scale_pos_weight=max(1.0, (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)),
    )
    model.fit(x_train, y_train)
    return model


def predict_baseline(model: XGBClassifier, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1]
