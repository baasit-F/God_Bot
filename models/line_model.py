"""Market-anchored margin and total models for spreads and totals.

Same idea as the moneyline model, in points instead of log-odds:

    E[margin] = spread_mu + f(SPREAD_FEATURES)
    E[total]  = total_mu  + f(TOTAL_FEATURES)

where `spread_mu` / `total_mu` are the means implied by the sharp line. f is a
ridge regression on the residual (actual - market mean), so with little data
or weak signal it collapses to the market. The outcome is then modelled as a
discretised Normal around that mean (see models.lines).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings
from models.features import LINE_FEATURES, LINE_PREFIX

TARGETS = {"spreads": "actual_margin", "totals": "actual_total"}


class OffsetRidge:
    """Ridge regression of y - offset on standardised X (intercept penalised too)."""

    def __init__(self, l2: float) -> None:
        self.l2 = l2
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.resid_sd_: float = float("nan")

    def _design(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(X)), (X - self.mean_) / self.scale_])

    def fit(self, X: np.ndarray, y: np.ndarray, offset: np.ndarray) -> "OffsetRidge":
        self.mean_ = X.mean(axis=0)
        self.scale_ = np.where(X.std(axis=0) > 1e-9, X.std(axis=0), 1.0)
        A = self._design(X)
        r = y - offset
        self.coef_ = np.linalg.solve(A.T @ A + self.l2 * np.eye(A.shape[1]), A.T @ r)
        self.resid_sd_ = float(np.std(r - A @ self.coef_, ddof=A.shape[1]))
        return self

    def predict(self, X: np.ndarray, offset: np.ndarray) -> np.ndarray:
        return offset + self._design(X) @ self.coef_


class LineModel:
    def __init__(self, market: str, l2: float | None = None) -> None:
        if market not in TARGETS:
            raise ValueError(f"{market!r} is not a line market")
        self.market = market
        self.features = LINE_FEATURES[market]
        self.offset = f"{LINE_PREFIX[market]}_mu"
        default = settings.spread_model_l2 if market == "spreads" else settings.total_model_l2
        self.l2 = default if l2 is None else l2
        self.model: OffsetRidge | None = None

    def usable(self, df: pd.DataFrame) -> pd.Series:
        return df[self.offset].notna()

    def fit(self, df: pd.DataFrame) -> "LineModel":
        train = df[self.usable(df) & df[TARGETS[self.market]].notna()]
        self.model = OffsetRidge(self.l2).fit(
            train[self.features].to_numpy(float), train[TARGETS[self.market]].to_numpy(float), train[self.offset].to_numpy(float)
        )
        return self

    def predict_mean(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        offset = df[self.offset].to_numpy(float)
        cap = settings.max_spread_dev if self.market == "spreads" else settings.max_total_dev
        adjust = self.model.predict(df[self.features].to_numpy(float), offset) - offset
        return offset + np.clip(adjust, -cap, cap)

    def coefficients(self) -> pd.Series | None:
        if self.model is None:
            return None
        return pd.Series(self.model.coef_[1:], index=self.features).sort_values(key=abs, ascending=False)
