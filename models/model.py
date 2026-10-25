"""Market-anchored win-probability models.

The de-vigged sharp line is the starting point, not just another feature:

    logit(p_home) = sharp_logit + f(other features)

so the model only learns *adjustments* to the market, and regularisation shrinks
it back toward the market rather than toward 50%. (A plain regularised logistic
regression on ~1,500 NFL games pulls every probability toward 0.5, which makes
underdogs look systematically +EV.)

Baseline: L2-penalised logistic regression with the market as a fixed offset,
fitted by Newton-Raphson. Optional: LightGBM boosting from the same offset
(`init_score`), if installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from config import settings
from models.features import FEATURES

OFFSET_FEATURE = "sharp_logit"


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class OffsetLogit:
    """Logistic regression with a fixed per-row offset and an L2 penalty on the slopes."""

    def __init__(self, l2: float = 1.0, max_iter: int = 50, tol: float = 1e-8) -> None:
        self.l2, self.max_iter, self.tol = l2, max_iter, tol
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None  # [intercept, slopes...] on standardised X

    def _design(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mean_) / self.scale_
        return np.column_stack([np.ones(len(Z)), Z])

    def fit(self, X: np.ndarray, y: np.ndarray, offset: np.ndarray) -> "OffsetLogit":
        self.mean_ = X.mean(axis=0)
        self.scale_ = np.where(X.std(axis=0) > 1e-9, X.std(axis=0), 1.0)
        A = self._design(X)
        beta = np.zeros(A.shape[1])
        # The intercept is penalised too: the market already prices home field.
        penalty = np.full(A.shape[1], self.l2)
        for _ in range(self.max_iter):
            p = _sigmoid(offset + A @ beta)
            grad = A.T @ (p - y) + penalty * beta
            hess = (A * (p * (1 - p))[:, None]).T @ A + np.diag(penalty)
            step = np.linalg.solve(hess, grad)
            beta -= step
            if np.max(np.abs(step)) < self.tol:
                break
        self.coef_ = beta
        return self

    def decision_function(self, X: np.ndarray, offset: np.ndarray) -> np.ndarray:
        return offset + self._design(X) @ self.coef_


class WinProbModel:
    def __init__(self, model_type: str | None = None, features: list[str] | None = None, seed: int | None = None) -> None:
        self.model_type = model_type or settings.model_type
        self.features = [f for f in (features or FEATURES) if f != OFFSET_FEATURE]
        self.seed = settings.random_seed if seed is None else seed
        self.model: OffsetLogit | object | None = None

    def fit(self, df: pd.DataFrame) -> "WinProbModel":
        train = df[df["home_win"].isin([0.0, 1.0])]  # ties carry no win/loss signal
        X, y = train[self.features].to_numpy(float), train["home_win"].to_numpy(float)
        offset = train[OFFSET_FEATURE].to_numpy(float)
        if self.model_type == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
            except ImportError as exc:
                raise ImportError("VP_MODEL=lightgbm requires `pip install lightgbm`") from exc
            self.model = LGBMClassifier(
                n_estimators=150, learning_rate=0.02, num_leaves=7, min_child_samples=60,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                random_state=self.seed, verbose=-1,
            ).fit(X, y.astype(int), init_score=offset)
        else:
            self.model = OffsetLogit(l2=settings.model_l2).fit(X, y, offset)
        return self

    def predict_home(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        X, offset = df[self.features].to_numpy(float), df[OFFSET_FEATURE].to_numpy(float)
        if isinstance(self.model, OffsetLogit):
            adjust = self.model.decision_function(X, offset) - offset
        else:
            adjust = self.model.predict(X, raw_score=True)
        cap = settings.max_logit_dev
        return _sigmoid(offset + np.clip(adjust, -cap, cap))

    def coefficients(self) -> pd.Series | None:
        """Standardised slopes of the market-offset logit (log-odds per 1 sd)."""
        if not isinstance(self.model, OffsetLogit):
            return None
        return pd.Series(self.model.coef_[1:], index=self.features).sort_values(key=abs, ascending=False)


def evaluate_probs(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1]))}
