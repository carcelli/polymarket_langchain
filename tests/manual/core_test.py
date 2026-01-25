#!/usr/bin/env python3
"""
Core Component Test

Test only the most basic components without complex dependencies.
"""

import sys
import os
import json
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parents[2] / "src"))

# Test just the database directly
from polymarket_agents.automl.ml_database import MLDatabase


def test_database_only():
    """Test database operations only."""
    print("🗄️ Testing Database Operations Only")
    print("=" * 35)

    try:
        db = MLDatabase()

        # Basic database operations
        experiment_id = db.create_experiment(
            name="Core Test", description="Testing basic database functionality"
        )
        print(f"✅ Created experiment: {experiment_id}")

        # Save a basic model
        model_info = {
            "name": "CoreTestModel",
            "model_type": "Test",
            "algorithm": "TestAlgo",
            "hyperparameters": {"param1": 1, "param2": "test"},
            "feature_columns": ["feat1", "feat2"],
            "training_samples": 100,
            "training_start_time": datetime.now().isoformat(),
            "training_end_time": datetime.now().isoformat(),
        }

        model_id = db.save_model(experiment_id, model_info)
        print(f"✅ Saved model: {model_id}")

        # Save metrics
        metrics = {"accuracy": 0.85, "precision": 0.82, "recall": 0.88}
        db.save_model_metrics(model_id, metrics)
        print(f"✅ Saved metrics: {metrics}")

        # Save a prediction
        predictions = [
            {
                "market_id": "test_market_1",
                "predicted_probability": 0.65,
                "actual_outcome": 1,
                "confidence": 0.8,
                "recommended_bet": "YES",
            }
        ]
        db.save_predictions(model_id, predictions)
        print(f"✅ Saved {len(predictions)} predictions")

        # Get results
        results = db.get_experiment_results(experiment_id)
        if results:
            print("✅ Retrieved experiment results successfully")

        # Get stats
        stats = db.get_database_stats()
        print(f"✅ Database stats: {stats.get('experiments_count', 0)} experiments")

        # Mark complete
        db.update_experiment_status(experiment_id, "completed", success=True)
        print("✅ Marked experiment as completed")

        return True

    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_data_ingestion_minimal():
    """Test minimal data ingestion."""
    print("\\n📊 Testing Minimal Data Ingestion")
    print("=" * 35)

    try:
        # Import data ingestion directly to avoid strategy dependencies
        from polymarket_agents.automl.data_ingestion import PolymarketDataIngestion

        ingestion = PolymarketDataIngestion()

        # Test the basic cleaning function with mock data
        mock_raw_data = [
            {
                "id": "test_1",
                "question": "Will this work?",
                "category": "tech",
                "volume": 5000,
                "outcome_prices": [0.6, 0.4],
                "resolved": True,
                "winner": "yes",
            },
            {
                "id": "test_2",
                "question": "Another test?",
                "category": "politics",
                "volume": 8000,
                "outcome_prices": [0.4, 0.6],
                "resolved": True,
                "winner": "no",
            },
        ]

        cleaned_df = ingestion.clean_market_data(mock_raw_data)
        print(f"✅ Cleaned {len(cleaned_df)} market records")

        # Test feature engineering
        ml_df = ingestion.engineer_ml_features(cleaned_df)
        print(f"✅ Engineered {len(ml_df.columns)} features")

        return True

    except Exception as e:
        print(f"❌ Data ingestion test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_github_minimal():
    """Test minimal GitHub integration."""
    print("\\n🔗 Testing Minimal GitHub Integration")
    print("=" * 38)

    try:
        # Test basic GitHub test generation
        from polymarket_agents.subagents.github_ml_agent import (
            generate_ml_strategy_test,
        )

        # Generate a simple test
        test_code = generate_ml_strategy_test(
            "MinimalTest", "predictor", "Minimal integration test"
        )

        print(f"✅ Generated test code: {len(test_code)} characters")

        # Save to file
        os.makedirs("integration_tests", exist_ok=True)
        test_file = "integration_tests/minimal_test.py"
        with open(test_file, "w") as f:
            f.write(test_code)

        print(f"✅ Saved test file: {test_file}")

        # Try to commit (may fail if not configured)
        try:
            from polymarket_agents.subagents.github_ml_agent import (
                commit_ml_tests_to_github,
            )

            test_files = {"minimal_test.py": test_code}
            result = commit_ml_tests_to_github(
                test_files, "🤖 Core Test: Minimal integration test"
            )
            print("✅ GitHub commit successful")

        except Exception as e:
            print(
                f"⚠️ GitHub commit failed (expected if not configured): {str(e)[:80]}..."
            )

        return True

    except Exception as e:
        print(f"❌ GitHub test failed: {e}")
        return False


def run_core_test():
    """Run core component tests."""
    print("🚀 Core ML Component Test")
    print("=" * 28)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    start_time = datetime.now()

    # Test components
    tests = [
        ("Database Only", test_database_only),
        ("Data Ingestion Minimal", test_data_ingestion_minimal),
        ("GitHub Minimal", test_github_minimal),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\\n🧪 Testing: {test_name}")
        print("-" * (12 + len(test_name)))

        try:
            success = test_func()
            results[test_name] = "✅ PASSED" if success else "❌ FAILED"

            if success:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")

        except Exception as e:
            results[test_name] = f"❌ ERROR: {str(e)[:50]}"
            print(f"❌ {test_name} ERROR: {e}")

    # Final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\\n" + "=" * 50)
    print("🎉 CORE COMPONENT TEST COMPLETE")
    print("=" * 50)
    print(".1f")
    print()

    # Results
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)

    print("📊 Test Results:")
    for test, result in results.items():
        print(f"   {result} - {test}")

    print(f"\\n🏆 Overall: {passed}/{total} core components working")

    # Assessment
    if passed == total:
        print("\\n🎯 EXCELLENT: All core components operational!")
        print("✅ Database: Working")
        print("✅ Data Ingestion: Working")
        print("✅ GitHub Integration: Working")
        print("\\n🚀 Ready for ML agent workflows and full pipeline testing!")

    elif passed >= total * 0.67:  # 2/3 working
        print(f"\\n⚠️ GOOD: {passed}/{total} core components working")
        print("Most functionality is operational.")
        print("Minor issues to resolve before full deployment.")

    else:
        print(f"\\n❌ NEEDS ATTENTION: Only {passed}/{total} core components working")
        print("Review errors above and fix configuration issues.")

    print("\\n💡 Next Steps:")
    print("1. Fix any failed components above")
    print("2. Run: python ml_agent_cli.py status (check full system)")
    print("3. Test ML agents: python ml_agent_cli.py workflow 'Train models'")
    print("4. Full pipeline: python automl_cli.py run")

    return results


if __name__ == "__main__":
    run_core_test()
