#!/usr/bin/env python3
"""
Complete ML Pipeline Demo for Sports Market Prediction

This script demonstrates the end-to-end machine learning pipeline:
1. Dataset preparation
2. Model training and evaluation
3. Live prediction on markets
4. Integration with planning agent

Usage:
    python demo_ml_pipeline.py
"""

import sys
from pathlib import Path

# Add agents directory to path
sys.path.append(str(Path(__file__).parents[1] / "src"))

from polymarket_agents.ml_strategies.xgboost_strategy import XGBoostProbabilityStrategy
import pandas as pd
import json


def demo_dataset_preparation():
    """Demo: Dataset preparation and exploration."""
    print("📊 Dataset Preparation Demo")
    print("=" * 40)

    # Load synthetic dataset
    dataset_path = Path("data/sports_ml_dataset_synthetic.parquet")
    if not dataset_path.exists():
        print(
            "❌ Dataset not found. Run: python scripts/python/prepare_sports_dataset.py"
        )
        return None

    df = pd.read_parquet(dataset_path)
    print(f"✅ Loaded {len(df)} sports markets")
    print(f"   Features: {len(df.columns)} columns")
    print(f"   Target balance: {df['target'].mean():.1%} YES outcomes")

    # Show sample data
    print("\n📋 Sample Markets:")
    sample = df[
        ["market_id", "question", "sport", "volume", "market_prob", "target"]
    ].head(3)
    for _, row in sample.iterrows():
        outcome = "YES" if row["target"] == 1 else "NO"
        print(
            f"   {row['question'][:50]}... | Volume: ${row['volume']:.0f} | "
            f"Outcome: {outcome}"
        )

    return df


def demo_model_training():
    """Demo: Model training and evaluation."""
    print("\n🤖 Model Training Demo")
    print("=" * 40)

    # Load dataset
    df = pd.read_parquet("data/sports_ml_dataset_synthetic.parquet")

    # Initialize and train model
    strategy = XGBoostProbabilityStrategy(name="demo_model")

    print("🚀 Training XGBoost model...")
    results = strategy.run_full_pipeline_from_data(df, test_size=0.2)

    # Display results
    eval_metrics = results["evaluation_metrics"]

    print("📊 Model Performance:")
    print(f"   Precision: {eval_metrics.get('precision', 0):.3f}")
    print(f"   Recall: {eval_metrics.get('recall', 0):.3f}")
    print(f"   Accuracy: {eval_metrics.get('accuracy', 0):.1%}")
    print(f"   F1 Score: {eval_metrics.get('f1', 0):.1%}")

    print("🎯 Edge Detection:")
    print(f"   Average Edge: {eval_metrics.get('avg_edge', 0):.1%}")
    print(f"   Edge Accuracy: {eval_metrics.get('edge_accuracy', 0):.1%}")

    # Show top features
    print("\n🔍 Top 5 Important Features:")
    importance = results["feature_importance"]
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (feature, score) in enumerate(sorted_features, 1):
        print(f"   {i}. {feature}: {score:.1%}")

    return strategy


def demo_live_prediction(strategy):
    """Demo: Making predictions on new markets."""
    print("\n🎯 Live Prediction Demo")
    print("=" * 40)

    # Example markets (simulating live data)
    test_markets = [
        {
            "market_id": "demo_nfl_market",
            "question": "Will the Kansas City Chiefs win Super Bowl 2026?",
            "sport": "NFL",
            "volume": 150000,
            "liquidity": 25000,
            "days_to_expiry": 60,
            "market_prob": 0.35,  # Current market price
            "category": "sports",
        },
        {
            "market_id": "demo_basketball_market",
            "question": "Will LeBron James score 30+ points in his next game?",
            "sport": "NBA",
            "volume": 75000,
            "liquidity": 15000,
            "days_to_expiry": 2,
            "market_prob": 0.65,
            "category": "sports",
        },
    ]

    for market in test_markets:
        print(f"\n🏈 Market: {market['question'][:60]}...")
        print(
            f"   Volume: ${market['volume']:.0f} | "
            f"Days to expiry: {market['days_to_expiry']}"
        )

        # Make prediction
        result = strategy.predict(market)

        print(f"   Model prediction: {result.probability:.3f}")
        print(f"   Market price: {market['market_prob']:.1%}")
        print(f"   Edge: {result.edge:.1%}")
        print(f"   Recommendation: {result.recommended_bet}")
        print(f"   Position size: {result.position_size:.3f}")
        print(f"   Confidence: {result.confidence:.1%}")


def demo_agent_integration():
    """Demo: Integration with planning agent."""
    print("\n🧠 Agent Integration Demo")
    print("=" * 40)

    print("When MARKET_FOCUS=sports, the planning agent will:")
    print("1. 🔍 Search only sports markets")
    print("2. 🤖 Use XGBoost predictions alongside LLM estimates")
    print("3. 📊 Calculate edges and position sizes")
    print("4. 💰 Make betting recommendations")

    print("\nTo test integration:")
    print("export MARKET_FOCUS=sports")
    print(
        "python scripts/python/cli.py run-memory-agent 'Find NFL markets with ML edge'"
    )

    print("\nThe agent will show XGBoost predictions in the reasoning!")


def main():
    """Run the complete ML pipeline demo."""
    print("🚀 Polymarket Sports ML Pipeline Demo")
    print("=" * 50)

    # Check for required files
    dataset_path = Path("data/sports_ml_dataset_synthetic.parquet")
    if not dataset_path.exists():
        print("⚠️  Dataset not found. Generating sample dataset...")
        print("Run: python scripts/python/prepare_sports_dataset.py")
        return

    # Demo 1: Dataset exploration
    df = demo_dataset_preparation()
    if df is None:
        return

    # Demo 2: Model training
    strategy = demo_model_training()

    # Demo 3: Live predictions
    demo_live_prediction(strategy)

    # Demo 4: Agent integration
    demo_agent_integration()

    print("\n🎉 Demo Complete!")
    print("\nNext Steps:")
    print("1. Train on real resolved markets when available")
    print("2. Fine-tune hyperparameters for better performance")
    print("3. Add time-series features (price momentum, volume trends)")
    print("4. Implement ensemble methods for robustness")
    print("5. Deploy in production with proper monitoring")


if __name__ == "__main__":
    main()
