#!/usr/bin/env python3
"""
Minimal ML Integration Test

Test only the core database and data ingestion components.
"""

import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parents[2] / "src"))

from polymarket_agents.automl.ml_database import MLDatabase


def test_database_core():
    """Test basic database operations."""
    print("🗄️ Testing ML Database Core")
    print("=" * 28)

    try:
        db = MLDatabase()

        # Create experiment
        experiment_id = db.create_experiment(
            name="Minimal Test", description="Testing core database functionality"
        )
        print(f"✅ Created experiment: {experiment_id}")

        # Create simple model entry
        model_info = {
            "name": "TestModel",
            "model_type": "Test",
            "algorithm": "Test",
            "hyperparameters": {"test": True},
            "feature_columns": ["test_feature"],
            "training_samples": 10,
            "training_start_time": datetime.now().isoformat(),
            "training_end_time": datetime.now().isoformat(),
        }

        model_id = db.save_model(experiment_id, model_info)
        print(f"✅ Saved model: {model_id}")

        # Save simple metrics
        metrics = {"accuracy": 0.8, "test_metric": 0.9}
        db.save_model_metrics(model_id, metrics)
        print(f"✅ Saved metrics: {metrics}")

        # Get results
        results = db.get_experiment_results(experiment_id)
        if results:
            print("✅ Retrieved experiment results")

        # Mark as completed
        db.update_experiment_status(experiment_id, "completed", success=True)
        print("✅ Updated experiment status")

        return True

    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_data_ingestion_core():
    """Test basic data ingestion."""
    print("\\n📊 Testing Data Ingestion Core")
    print("=" * 32)

    try:
        from polymarket_agents.automl.data_ingestion import PolymarketDataIngestion

        ingestion = PolymarketDataIngestion()

        # Create minimal mock data
        import pandas as pd
        import numpy as np

        np.random.seed(42)
        mock_data = []
        for i in range(10):
            mock_data.append(
                {
                    "market_id": f"mock_{i}",
                    "question": f"Question {i}?",
                    "category": "test",
                    "volume": 1000 + i * 100,
                    "yes_price": 0.5 + np.random.random() * 0.4,
                    "no_price": 0.5 - np.random.random() * 0.4,
                    "resolved": True,
                    "actual_outcome": np.random.choice([0, 1]),
                }
            )

        mock_df = pd.DataFrame(mock_data)
        print(f"✅ Created mock data: {len(mock_df)} samples")

        # Test cleaning
        cleaned_df = ingestion.clean_market_data(mock_data)
        print(f"✅ Cleaned data: {len(cleaned_df)} samples")

        # Test feature engineering
        ml_df = ingestion.engineer_ml_features(cleaned_df)
        print(f"✅ Engineered features: {len(ml_df.columns)} columns")

        return True

    except Exception as e:
        print(f"❌ Data ingestion test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_github_basic():
    """Test basic GitHub integration."""
    print("\\n🔗 Testing GitHub Integration")
    print("=" * 30)

    try:
        # Test basic GitHub agent import and test generation
        from polymarket_agents.subagents.github_ml_agent import (
            generate_ml_strategy_test,
        )

        test_code = generate_ml_strategy_test(
            "BasicTest", "predictor", "Basic integration test"
        )

        print(f"✅ Generated test code: {len(test_code)} characters")

        # Save test file
        import os

        test_dir = "integration_tests"
        os.makedirs(test_dir, exist_ok=True)

        test_file = os.path.join(test_dir, "basic_test.py")
        with open(test_file, "w") as f:
            f.write(test_code)

        print(f"✅ Saved test file: {test_file}")

        return True

    except Exception as e:
        print(f"❌ GitHub test failed: {e}")
        return False


def run_minimal_test():
    """Run minimal integration test."""
    print("🚀 Minimal ML Integration Test")
    print("=" * 35)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    start_time = datetime.now()
    tests = [
        ("Database Core", test_database_core),
        ("Data Ingestion Core", test_data_ingestion_core),
        ("GitHub Basic", test_github_basic),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\\n🧪 Running: {test_name}")
        print("-" * (15 + len(test_name)))

        try:
            success = test_func()
            results[test_name] = "✅ PASSED" if success else "❌ FAILED"
        except Exception as e:
            results[test_name] = f"❌ ERROR: {str(e)[:50]}"
            print(f"❌ {test_name} failed: {e}")

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\\n" + "=" * 50)
    print("🎉 MINIMAL INTEGRATION TEST COMPLETE")
    print("=" * 50)
    print(".1f")
    print()

    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)

    print("📊 Results:")
    for test, result in results.items():
        print(f"   {result} - {test}")

    print(f"\\n🏆 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("\\n🎯 SUCCESS: Core ML components are working!")
        print("Ready to build more complex ML workflows.")
    else:
        print(f"\\n⚠️ PARTIAL SUCCESS: {passed}/{total} components working")
        print("Review failed tests before proceeding.")

    return results


if __name__ == "__main__":
    run_minimal_test()
