"""Win-probability models with probability calibration.

Baseline: standardised logistic regression wrapped in CalibratedClassifierCV.
Optional: LightGBM (if installed) with the same calibration wrapper. The market
probability is itself a feature, so the model learns *where to deviate* from the
sharp line rather than re-deriving the line from scratch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import settings
from models.features import FEATURES


class WinProbModel:
    def __init__(self, model_type: str | None = None, features: list[str] | None = None, seed: int | None = None) -> None:
        self.model_type = model_type or settings.model_type
        self.features = features or FEATURES
        self.seed = settings.random_seed if seed is None else seed
        self.model: CalibratedClassifierCV | None = None

    def _base_estimator(self):
        if self.model_type == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
            except ImportError as exc:
                raise ImportError("VP_MODEL=lightgbm requires `pip install lightgbm`") from exc
            return LGBMClassifier(
                n_estimators=300,
                learning_rate=0.02,
                num_leaves=15,
                min_child_samples=40,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=self.seed,
                verbose=-1,
            )
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))

    def fit(self, df: pd.DataFrame) -> "WinProbModel":
        train = df.dropna(subset=["home_win"])
        X, y = train[self.features].to_numpy(), train["home_win"].astype(int).to_numpy()
        # Time-ordered folds would be stricter; calibration folds only rescale
        # probabilities and all rows are already strictly before the test date.
        self.model = CalibratedClassifierCV(self._base_estimator(), method="sigmoid", cv=3)
        self.model.fit(X, y)
        return self

    def predict_home(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        return self.model.predict_proba(df[self.features].to_numpy())[:, 1]

    def coefficients(self) -> pd.Series | None:
        """Average standardised LR coefficients across calibration folds."""
        if self.model is None or self.model_type != "logistic":
            return None
        coefs = [cc.estimator.named_steps["logisticregression"].coef_[0] for cc in self.model.calibrated_classifiers_]
        return pd.Series(np.mean(coefs, axis=0), index=self.features).sort_values(key=abs, ascending=False)


def evaluate_probs(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1]))}
