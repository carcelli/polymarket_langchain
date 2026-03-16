"""
Bootstrap XGBoost Probability Model

Generates synthetic-but-statistically-calibrated training data and trains a
baseline XGBoost model when no real historical data is available.

The synthetic data replicates Polymarket market structure (prices, volumes,
category flags, resolution outcomes) so the trained model can be immediately
used for inference while real data accumulates.

Usage:
    python scripts/python/bootstrap_xgboost.py
    python scripts/python/bootstrap_xgboost.py --samples 2000 --output data/models/xgboost_probability_model.json
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Make src importable from scripts directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

CATEGORIES = ["politics", "sports", "crypto", "geopolitics", "tech", "economics", "other"]
VOLUME_BINS = [0, 1000, 10000, 100_000, 1_000_000, float("inf")]
VOLUME_LABELS = ["micro", "small", "medium", "large", "huge"]

RNG = np.random.default_rng(42)


def _sample_yes_price(n: int) -> np.ndarray:
    """
    Sample market prices from a realistic bimodal distribution.

    Real Polymarket prices cluster near 0 and 1 (decisive markets) with
    a secondary mass around 0.5 (contested markets).
    """
    # 40% contested (near 0.5), 60% decisive (near extremes)
    contested_mask = RNG.random(n) < 0.40
    contested = RNG.beta(5, 5, size=n)          # tight bell around 0.5
    decisive = RNG.beta(0.6, 0.6, size=n)       # U-shaped toward extremes
    prices = np.where(contested_mask, contested, decisive)
    return np.clip(prices, 0.02, 0.98)


def _resolution_from_price(yes_price: np.ndarray) -> np.ndarray:
    """
    Simulate resolution: market is well-calibrated but noisy.

    P(resolve YES | price=p) = p + noise, clipped to [0,1].
    """
    noise = RNG.normal(0, 0.08, size=len(yes_price))
    true_prob = np.clip(yes_price + noise, 0.0, 1.0)
    return RNG.binomial(1, true_prob).astype(int)


def generate_synthetic_dataset(n_samples: int = 1500) -> pd.DataFrame:
    """
    Generate n_samples synthetic market records that match the feature schema
    expected by XGBoostProbabilityStrategy._prepare_features_for_training().
    """
    logger.info(f"Generating {n_samples} synthetic training samples...")

    yes_price = _sample_yes_price(n_samples)
    no_price = 1 - yes_price

    volume = np.exp(RNG.uniform(np.log(500), np.log(5_000_000), n_samples))
    liquidity = volume * RNG.uniform(0.05, 0.8, n_samples)
    volume_to_liquidity = volume / np.maximum(liquidity, 1)

    category_idx = RNG.integers(0, len(CATEGORIES), n_samples)
    category = np.array(CATEGORIES)[category_idx]

    question_length = RNG.integers(20, 120, n_samples)
    description_length = RNG.integers(0, 500, n_samples)
    has_description = description_length > 0

    market_age_days = RNG.integers(1, 365, n_samples).astype(float)
    days_until_end = RNG.integers(0, 180, n_samples).astype(float)
    days_to_resolve = market_age_days + days_until_end

    df = pd.DataFrame(
        {
            # Core price features
            "yes_price": yes_price,
            "no_price": no_price,
            "implied_probability": yes_price,
            "price_distance_from_fair": np.abs(yes_price - 0.5),
            "price_volatility": yes_price * (1 - yes_price),
            "price_extremity": ((yes_price < 0.2) | (yes_price > 0.8)).astype(int),
            "spread": np.abs(yes_price - no_price),
            "market_efficiency_score": 1 / (np.abs(yes_price - no_price) + 0.01),
            # Volume / liquidity
            "volume": volume,
            "liquidity": liquidity,
            "volume_to_liquidity": volume_to_liquidity,
            "log_volume": np.log(volume + 1),
            "volume_category": pd.Categorical(
                pd.cut(volume, bins=VOLUME_BINS, labels=VOLUME_LABELS),
                categories=VOLUME_LABELS,
            ),
            # Time features
            "market_age_days": market_age_days,
            "days_until_end": days_until_end,
            "days_to_resolve": days_to_resolve,
            # Text / metadata features
            "question_length": question_length.astype(float),
            "description_length": description_length.astype(float),
            "has_description": has_description.astype(int),
            "question_word_count": (question_length / 6).astype(int).astype(float),
            # Category flags
            "is_politics": (category == "politics").astype(int),
            "is_sports": (category == "sports").astype(int),
            "is_crypto": (category == "crypto").astype(int),
            "is_geopolitics": (category == "geopolitics").astype(int),
            "is_tech": (category == "tech").astype(int),
            "is_economics": (category == "economics").astype(int),
            # Keyword flags (correlated with category)
            "has_political_keywords": (category == "politics").astype(int),
            "has_sports_keywords": (category == "sports").astype(int),
            "has_crypto_keywords": (category == "crypto").astype(int),
            # Momentum placeholders (zero for baseline model)
            "price_momentum_24h": RNG.normal(0, 0.02, n_samples),
            "volume_trend_7d": RNG.normal(0, 0.1, n_samples),
            # Metadata (excluded from features by XGBoostProbabilityStrategy)
            "market_id": [f"synthetic_{i:05d}" for i in range(n_samples)],
            "question": [f"Synthetic question {i}" for i in range(n_samples)],
            "description": ["" for _ in range(n_samples)],
            "category": category,
            "active": True,
            "resolved": True,
            "actual_outcome": _resolution_from_price(yes_price),
            # Target column expected by _prepare_features_for_training
            "will_resolve_yes": _resolution_from_price(yes_price),
        }
    )

    pos_rate = df["will_resolve_yes"].mean()
    logger.info(
        f"Synthetic dataset ready: {len(df)} rows, {len(df.columns)} cols, "
        f"YES rate = {pos_rate:.1%}"
    )
    return df


def bootstrap_model(
    n_samples: int = 1500,
    output_path: str = "data/models/xgboost_probability_model.json",
) -> dict:
    """Train and save a baseline XGBoost model using synthetic data."""
    from polymarket_agents.ml_strategies.xgboost_strategy import (
        XGBoostProbabilityStrategy,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    training_data = generate_synthetic_dataset(n_samples)

    strategy = XGBoostProbabilityStrategy(
        name="xgboost_probability_baseline",
        model_path=output_path,
    )

    logger.info("Training baseline XGBoost model...")
    results = strategy.run_full_pipeline_from_data(training_data, test_size=0.2)

    metrics = results.get("evaluation_metrics", {})
    logger.info("=" * 55)
    logger.info("BASELINE MODEL RESULTS")
    logger.info("=" * 55)
    logger.info(f"  Accuracy : {metrics.get('accuracy', 0):.3f}")
    logger.info(f"  ROC-AUC  : {metrics.get('roc_auc', 0):.3f}")
    logger.info(f"  Log-loss : {metrics.get('log_loss', 0):.4f}")
    logger.info(f"  Brier    : {metrics.get('brier_score', 0):.4f}")
    logger.info(f"  Pos-edge : {metrics.get('positive_edge_pct', 0):.1%}")
    logger.info("=" * 55)
    logger.info(f"Model saved → {out}")

    # Also save to the canonical path the planning_agent looks for
    canonical = Path("data/models/xgboost_probability_model.json")
    if canonical != out and not canonical.exists():
        canonical.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(str(out), str(canonical))
        logger.info(f"Copied to canonical path → {canonical}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Bootstrap XGBoost probability model")
    parser.add_argument("--samples", type=int, default=1500, help="Synthetic sample count")
    parser.add_argument(
        "--output",
        type=str,
        default="data/models/xgboost_probability_model.json",
        help="Output model path",
    )
    args = parser.parse_args()

    os.chdir(Path(__file__).parent.parent.parent)  # project root
    bootstrap_model(n_samples=args.samples, output_path=args.output)


if __name__ == "__main__":
    main()
