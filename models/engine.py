"""One object that trains and scores every enabled market (h2h, spreads, totals).

Shared by the walk-forward backtester and the daily picks engine so both apply
exactly the same models and rules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings
from models.features import LINE_PREFIX
from models.line_model import LineModel
from models.model import WinProbModel
from models.scoring import build_line_opportunities, build_opportunities

MARKET_LABELS = {"h2h": "Moneyline", "spreads": "Spread", "totals": "Total"}


class MarketModels:
    def __init__(self, markets: list[str] | None = None, model_type: str | None = None) -> None:
        self.markets = markets or settings.markets
        self.model_type = model_type
        self.h2h: WinProbModel | None = None
        self.lines: dict[str, LineModel] = {}

    def fit(self, history: pd.DataFrame) -> "MarketModels":
        if "h2h" in self.markets:
            self.h2h = WinProbModel(self.model_type).fit(history[history["has_h2h"]])
        for m in ("spreads", "totals"):
            if m in self.markets:
                lm = LineModel(m)
                if (lm.usable(history) & history[f"actual_{'margin' if m == 'spreads' else 'total'}"].notna()).sum() >= 50:
                    self.lines[m] = lm.fit(history)
        return self

    def opportunities(self, slate: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
        parts = []
        if self.h2h is not None:
            rows = slate[slate["has_h2h"]]
            if not rows.empty:
                parts.append(build_opportunities(rows, self.h2h.predict_home(rows)))
        for m, model in self.lines.items():
            rows = slate[model.usable(slate)]
            if not rows.empty:
                parts.append(build_line_opportunities(rows, quotes, model.predict_mean(rows), m))
        parts = [p for p in parts if not p.empty]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def predictions(self, slate: pd.DataFrame) -> pd.DataFrame:
        """Model vs market per game, for out-of-sample calibration reports."""
        out = slate[["game_id", "game_date", "home_win", "p_sharp_home", "actual_margin", "actual_total",
                     "spread_mu", "total_mu"]].copy()
        out["p_model_home"] = np.nan
        if self.h2h is not None and slate["has_h2h"].any():
            out.loc[slate["has_h2h"].values, "p_model_home"] = self.h2h.predict_home(slate[slate["has_h2h"]])
        for m, model in self.lines.items():
            col = f"{LINE_PREFIX[m]}_mu_model"
            out[col] = np.nan
            mask = model.usable(slate).values
            if mask.any():
                out.loc[mask, col] = model.predict_mean(slate[mask])
        return out.rename(columns={"p_sharp_home": "p_market_home"})
