"""Central configuration for ValuePicks Bot.

Every value can be overridden with an environment variable (or a `.env` file in
the project root). Thresholds live here so the backtester and the daily picks
engine always apply exactly the same rules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader so python-dotenv is not a hard dependency."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(PROJECT_ROOT / ".env")


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _env_list(key: str, default: list[str]) -> list[str]:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    # --- Storage -----------------------------------------------------------
    db_path: Path = field(default_factory=lambda: Path(_env("VP_DB_PATH", str(PROJECT_ROOT / "storage" / "valuepicks.db"))))
    parquet_dir: Path = field(default_factory=lambda: Path(_env("VP_PARQUET_DIR", str(PROJECT_ROOT / "storage" / "parquet"))))
    reports_dir: Path = field(default_factory=lambda: Path(_env("VP_REPORTS_DIR", str(PROJECT_ROOT / "reports"))))

    # --- API keys ----------------------------------------------------------
    odds_api_key: str = field(default_factory=lambda: _env("ODDS_API_KEY", ""))
    odds_api_base_url: str = field(default_factory=lambda: _env("ODDS_API_BASE_URL", "https://api.the-odds-api.com/v4"))
    sportsdataio_api_key: str = field(default_factory=lambda: _env("SPORTSDATAIO_API_KEY", ""))
    sentiment_api_key: str = field(default_factory=lambda: _env("SENTIMENT_API_KEY", ""))

    # --- Market ------------------------------------------------------------
    sport_key: str = field(default_factory=lambda: _env("VP_SPORT_KEY", "basketball_nba"))
    regions: str = field(default_factory=lambda: _env("VP_REGIONS", "us,eu"))
    # Sharp book used as the "true" market reference for de-vigging.
    sharp_book: str = field(default_factory=lambda: _env("VP_SHARP_BOOK", "pinnacle"))
    # Books we are actually able to place bets at (line shopping universe).
    bettable_books: list[str] = field(
        default_factory=lambda: _env_list("VP_BETTABLE_BOOKS", ["draftkings", "fanduel", "betmgm", "caesars"])
    )
    devig_method: str = field(default_factory=lambda: _env("VP_DEVIG_METHOD", "shin"))  # "shin" | "proportional"

    # --- Pick filters ------------------------------------------------------
    min_ev: float = field(default_factory=lambda: _env_float("VP_MIN_EV", 0.025))  # +2.5%
    min_vos: float = field(default_factory=lambda: _env_float("VP_MIN_VOS", 1.2))
    # The relative edge term of VOS is expressed in percent (x100). With a raw
    # fraction, a 10% edge at even odds scores 0.03 and VOS > 1.2 would need an
    # implausible 400% edge. See README "VOS scaling" for the details.
    vos_edge_scale: float = field(default_factory=lambda: _env_float("VP_VOS_EDGE_SCALE", 100.0))
    min_odds: float = field(default_factory=lambda: _env_float("VP_MIN_ODDS", 1.30))
    max_odds: float = field(default_factory=lambda: _env_float("VP_MAX_ODDS", 6.00))
    max_picks_per_day: int = field(default_factory=lambda: _env_int("VP_MAX_PICKS_PER_DAY", 10))

    # --- Bankroll / sizing -------------------------------------------------
    starting_bankroll: float = field(default_factory=lambda: _env_float("VP_BANKROLL", 10_000.0))
    kelly_fraction: float = field(default_factory=lambda: _env_float("VP_KELLY_FRACTION", 0.25))
    max_stake_pct: float = field(default_factory=lambda: _env_float("VP_MAX_STAKE_PCT", 0.03))
    max_daily_exposure_pct: float = field(default_factory=lambda: _env_float("VP_MAX_DAILY_EXPOSURE_PCT", 0.15))

    # --- Modelling / walk-forward -----------------------------------------
    # Shrink the model toward the sharp market in logit space:
    # p = market + w * (model - market). 1.0 trusts the model fully; lower values
    # counter the winner's curse of only betting where the model disagrees most.
    model_weight: float = field(default_factory=lambda: _env_float("VP_MODEL_WEIGHT", 0.5))
    model_type: str = field(default_factory=lambda: _env("VP_MODEL", "logistic"))  # "logistic" | "lightgbm"
    min_train_games: int = field(default_factory=lambda: _env_int("VP_MIN_TRAIN_GAMES", 400))
    retrain_every_days: int = field(default_factory=lambda: _env_int("VP_RETRAIN_DAYS", 14))
    # Hours before tip-off at which the bot "decides". Only odds snapshots at
    # or before this point are visible to the model (no closing-line leakage).
    decision_hours_before: float = field(default_factory=lambda: _env_float("VP_DECISION_HOURS", 3.0))
    random_seed: int = field(default_factory=lambda: _env_int("VP_SEED", 42))


settings = Settings()
