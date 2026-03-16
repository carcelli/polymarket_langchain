#!/usr/bin/env python3
"""
Train and Evaluate XGBoost Probability Calibration Strategy

This script demonstrates the complete ML pipeline for training an XGBoost model
to predict market resolution probabilities and identify betting edges.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import logging
import click

# Add the agents directory to path
sys.path.append(str(Path(__file__).parents[2] / "src"))

from polymarket_agents.ml_strategies.xgboost_strategy import XGBoostProbabilityStrategy

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--dataset",
    default="data/ml_training_dataset.parquet",
    help="Path to training dataset",
)
@click.option(
    "--model-name",
    default="xgboost_probability_model",
    help="Name for the trained model",
)
@click.option("--test-size", default=0.2, help="Fraction of data for testing")
@click.option(
    "--use-existing",
    is_flag=True,
    help="Use existing dataset instead of creating new one",
)
def train(dataset, model_name, test_size, use_existing):
    """Train XGBoost probability calibration model on sports markets."""
    print("🤖 XGBoost Probability Calibration Strategy Training")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Model: {model_name}")
    print(f"Test size: {test_size}")

    # Initialize strategy
    strategy = XGBoostProbabilityStrategy(
        name=model_name, model_path=f"data/models/{model_name}.json"
    )

    try:
        if use_existing and Path(dataset).exists():
            # Load existing dataset
            print(f"\n📂 Loading existing dataset from {dataset}...")
            import pandas as pd

            training_data = pd.read_parquet(dataset)
            print(f"Loaded {len(training_data)} samples")
        else:
            # Create new dataset
            print("\n🏗️  Creating training dataset...")
            training_data = strategy.create_training_dataset(
                days_back=365, min_volume=1000
            )

        if len(training_data) == 0:
            print("❌ No training data available!")
            return

        # Run the training pipeline
        print("\n🚀 Starting ML training pipeline...")
        results = strategy.run_full_pipeline_from_data(
            training_data=training_data, test_size=test_size
        )

        # Print results
        print("\n📊 Model Evaluation Results:")
        print("-" * 40)
        eval_metrics = results["evaluation_metrics"]
        print(f"  Accuracy:    {eval_metrics.get('accuracy', 0):.3f}")
        print(f"  Precision:   {eval_metrics.get('precision', 0):.3f}")
        print(f"  Recall:      {eval_metrics.get('recall', 0):.3f}")
        print(f"  F1 Score:    {eval_metrics.get('f1_score', 0):.3f}")
        print(f"  ROC-AUC:     {eval_metrics.get('roc_auc', 0):.3f}")
        print(f"  Brier Score: {eval_metrics.get('brier_score', 0):.3f}")

        print("\n💰 Backtest Results:")
        print("-" * 40)
        backtest = results["backtest_results"]
        print(f"  Initial capital: ${backtest.get('initial_capital', 0):,.1f}")
        print(f"  Final capital:   ${backtest.get('final_capital', 0):,.1f}")
        print(f"  Number of trades: {backtest.get('num_trades', 0)}")
        print(f"  Win rate:        {backtest.get('win_rate', 0):.1%}")
        print(f"  Sharpe ratio:    {backtest.get('sharpe_ratio', 0):.2f}")
        print(f"  Max drawdown:    {backtest.get('max_drawdown', 0):.1%}")

        print("\n🎯 Top 10 Most Important Features:")
        print("-" * 40)
        feature_importance = results["feature_importance"]
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:10]
        for i, (feature, importance) in enumerate(sorted_features, 1):
            print(f"  {i:2d}. {feature}: {importance:.4f}")

        # Save results to file
        results_file = Path("data/models/xgboost_training_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n💾 Results saved to {results_file}")
        print(f"🤖 Model saved to {results['model_path']}")

        # Success message
        roc_auc = eval_metrics.get("roc_auc", 0)
        total_return = backtest.get("total_return_pct", 0)

        if roc_auc > 0.7 and total_return > 5:
            print("\n🎉 Excellent results! Model shows strong predictive power.")
        elif roc_auc > 0.6 and total_return > 0:
            print("\n👍 Good results! Model shows promising predictive ability.")
        else:
            print(
                "\n🤔 Model needs improvement. Consider more data or feature engineering."
            )

        print("\n📈 Next Steps:")
        print(
            "1. Run prediction on live markets: python scripts/python/cli.py run-memory-agent 'Find markets with XGBoost edge'"
        )
        print("2. Monitor performance in production")
        print("3. Retrain periodically with new resolved markets")
        print("4. Experiment with hyperparameters for better performance")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\n❌ Training failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check that you have resolved market data in data/markets.db")
        print("2. Ensure you have at least 100 resolved markets with volume > $1000")
        print("3. Try reducing min_volume parameter if no data is found")
        sys.exit(1)


def quick_test():
    """Quick test of the strategy with minimal data."""
    print("🧪 Quick Test Mode")
    print("=" * 30)

    try:
        # Create minimal training data for testing
        import pandas as pd
        import numpy as np

        # Generate synthetic data for testing
        np.random.seed(42)
        n_samples = 200

        synthetic_data = pd.DataFrame(
            {
                "market_id": [f"market_{i}" for i in range(n_samples)],
                "question": [f"Test question {i}" for i in range(n_samples)],
                "category": np.random.choice(
                    ["sports", "politics", "crypto"], n_samples
                ),
                "yes_price": np.random.beta(
                    2, 2, n_samples
                ),  # Realistic probability distribution
                "no_price": lambda df: 1 - df["yes_price"],
                "volume": np.random.exponential(50000, n_samples) + 1000,
                "liquidity": np.random.exponential(10000, n_samples),
                "days_to_resolve": np.random.randint(1, 365, n_samples),
                "price_distance_from_fair": lambda df: abs(df["yes_price"] - 0.5),
                "log_volume": lambda df: np.log(df["volume"] + 1),
                "will_resolve_yes": np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
            }
        )

        # Fill in computed columns
        synthetic_data["no_price"] = 1 - synthetic_data["yes_price"]
        synthetic_data["price_distance_from_fair"] = abs(
            synthetic_data["yes_price"] - 0.5
        )
        synthetic_data["log_volume"] = np.log(synthetic_data["volume"] + 1)

        print(f"Created {len(synthetic_data)} synthetic training samples")

        # Train model
        strategy = XGBoostProbabilityStrategy(
            name="xgboost_test", model_path="data/models/xgboost_test_model.json"  # .json for XGBoost native format
        )

        print("Training model on synthetic data...")
        strategy.train(synthetic_data)

        # Test prediction - use same feature set as training
        test_market = {
            "market_id": "test_market",
            "question": "Will this test market resolve YES?",
            "yes_price": 0.6,
            "no_price": 0.4,
            "volume": 50000,
            "liquidity": 10000,
            "days_to_resolve": 30,
            "price_distance_from_fair": abs(0.6 - 0.5),
            "log_volume": np.log(50000 + 1),
            "category": "sports",
        }

        result = strategy.predict(test_market)
        print("\nSample prediction:")
        print(f"Predicted probability: {result.predicted_probability:.3f}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Recommendation: {result.recommended_bet}")
        print(f"Position size: {result.position_size:.3f}")
        print(f"Edge: {result.edge:.1%}")

        print("\n✅ Quick test successful!")

    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        quick_test()
    else:
        train()
