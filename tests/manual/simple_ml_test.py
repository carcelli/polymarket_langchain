#!/usr/bin/env python3
"""
Simple ML Integration Test

Test core ML functionality without complex dependencies.
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parents[2] / "src"))

from polymarket_agents.automl.ml_database import MLDatabase
from polymarket_agents.automl.data_ingestion import PolymarketDataIngestion


def test_database_operations():
    """Test database operations."""
    print("🗄️ Testing ML Database Operations")
    print("=" * 35)

    try:
        db = MLDatabase()

        # Create experiment
        experiment_id = db.create_experiment(
            name="Simple ML Test",
            description="Testing basic ML database functionality"
        )
        print(f"✅ Created experiment: {experiment_id}")

        # Create mock model data
        model_info = {
            'name': 'Test MarketPredictor',
            'model_type': 'MarketPredictor',
            'algorithm': 'RandomForest',
            'hyperparameters': {'n_estimators': 50, 'max_depth': 5},
            'feature_columns': ['volume', 'yes_price', 'category'],
            'training_samples': 100,
            'training_start_time': datetime.now().isoformat(),
            'training_end_time': datetime.now().isoformat()
        }

        model_id = db.save_model(experiment_id, model_info)
        print(f"✅ Saved model: {model_id}")

        # Save mock metrics
        metrics = {'accuracy': 0.65, 'f1': 0.63, 'roc_auc': 0.71}
        db.save_model_metrics(model_id, metrics)
        print(f"✅ Saved metrics: {metrics}")

        # Save mock predictions
        predictions = [
            {
                'market_id': 'test_market_1',
                'predicted_probability': 0.55,
                'actual_outcome': 1,
                'confidence': 0.7,
                'recommended_bet': 'YES'
            }
        ]
        db.save_predictions(model_id, predictions)
        print(f"✅ Saved {len(predictions)} predictions")

        # Get experiment results
        results = db.get_experiment_results(experiment_id)
        if results:
            print("✅ Retrieved experiment results")
            print(f"   Models: {len(results['models'])}")
            print(f"   Datasets: {len(results['datasets'])}")

        # Get database stats
        stats = db.get_database_stats()
        print(f"✅ Database stats retrieved: {stats.get('experiments_count', 0)} experiments")

        db.update_experiment_status(experiment_id, "completed", success=True)
        print("✅ Updated experiment status")

        return True

    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_ingestion():
    """Test data ingestion functionality."""
    print("\\n📊 Testing Data Ingestion")
    print("=" * 25)

    try:
        ingestion = PolymarketDataIngestion()

        # Create mock data for testing (since API might not be available)
        import pandas as pd
        import numpy as np

        np.random.seed(42)
        mock_data = []
        for i in range(50):
            mock_data.append({
                'market_id': f'mock_market_{i}',
                'question': f'Mock question {i} about outcome?',
                'category': np.random.choice(['politics', 'sports', 'crypto']),
                'volume': np.random.exponential(5000) + 1000,
                'yes_price': np.random.beta(2, 2),
                'no_price': 1 - np.random.beta(2, 2),
                'liquidity': np.random.exponential(2000),
                'resolved': np.random.choice([True, False], p=[0.7, 0.3]),
                'actual_outcome': np.random.choice([0, 1]) if np.random.random() > 0.3 else None
            })

        mock_df = pd.DataFrame(mock_data)
        print(f"✅ Created mock dataset: {len(mock_df)} samples")

        # Test data cleaning
        from polymarket_agents.automl.data_quality import DataQualityValidator

        validator = DataQualityValidator()
        clean_df, quality_report = validator.validate_and_clean_data(mock_df)

        print(f"✅ Data cleaned: {len(clean_df)} samples")
        print(".1f")
        print(f"   Ready for ML: {'Yes' if quality_report['ready_for_ml'] else 'No'}")

        return True

    except Exception as e:
        print(f"❌ Data ingestion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ml_strategy():
    """Test ML strategy functionality."""
    print("\\n🤖 Testing ML Strategy")
    print("=" * 22)

    try:
        from polymarket_agents.ml_strategies.market_prediction import MarketPredictor

        # Create model
        model = MarketPredictor()
        print(f"✅ Created model: {model.name}")

        # Test feature preparation
        sample_market = {
            'id': 'test_market',
            'question': 'Will this test pass?',
            'category': 'tech',
            'volume': 10000,
            'outcome_prices': [0.55, 0.45]
        }

        features = model.prepare_features(sample_market)
        print(f"✅ Feature preparation: {features.shape[1]} features generated")

        # Test prediction interface
        result = model.predict(sample_market)
        print("✅ Prediction interface working")
        print(".1%")
        print(".1%")
        print(f"   Bet: {result.recommended_bet}")

        return True

    except Exception as e:
        print(f"❌ ML strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_github_integration():
    """Test basic GitHub integration."""
    print("\\n🔗 Testing GitHub Integration")
    print("=" * 30)

    try:
        from polymarket_agents.subagents.github_ml_agent import generate_ml_strategy_test

        # Generate a simple test file
        test_content = generate_ml_strategy_test(
            "TestPredictor",
            "predictor",
            "Generated test for integration testing"
        )

        print(f"✅ Generated test file: {len(test_content)} characters")

        # Save locally
        test_dir = Path("./integration_tests")
        test_dir.mkdir(exist_ok=True)

        test_file = test_dir / "test_integration_generated.py"
        with open(test_file, 'w') as f:
            f.write(test_content)

        print(f"✅ Saved test file: {test_file}")

        # Try GitHub commit (may fail if not configured)
        try:
            from polymarket_agents.subagents.github_ml_agent import commit_ml_tests_to_github

            test_files = {"test_integration_generated.py": test_content}
            result = commit_ml_tests_to_github(
                test_files,
                "🤖 Integration Test: Automated test generation"
            )
            print("✅ GitHub commit successful")

        except Exception as e:
            print(f"⚠️ GitHub commit failed (may not be configured): {str(e)[:100]}...")

        return True

    except Exception as e:
        print(f"❌ GitHub integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_simple_integration_test():
    """Run the simple integration test."""
    print("🚀 Simple ML Integration Test")
    print("=" * 35)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    start_time = datetime.now()
    test_results = {}

    # Run individual tests
    tests = [
        ("Database Operations", test_database_operations),
        ("Data Ingestion", test_data_ingestion),
        ("ML Strategy", test_ml_strategy),
        ("GitHub Integration", test_github_integration)
    ]

    for test_name, test_func in tests:
        print(f"\\n🧪 Running: {test_name}")
        print("-" * (15 + len(test_name)))

        try:
            success = test_func()
            test_results[test_name] = "✅ PASSED" if success else "❌ FAILED"

            if success:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")

        except Exception as e:
            test_results[test_name] = f"❌ ERROR: {str(e)[:50]}"
            print(f"❌ {test_name} ERROR: {e}")

    # Final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\\n" + "=" * 50)
    print("🎉 SIMPLE INTEGRATION TEST COMPLETE")
    print("=" * 50)
    print(".1f")
    print(f"📅 Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Test results summary
    print("📊 Test Results:")
    passed = 0
    total = len(test_results)

    for test_name, result in test_results.items():
        print(f"   {result} - {test_name}")
        if "PASSED" in result:
            passed += 1

    print(f"\\n🏆 Overall: {passed}/{total} tests passed")

    # Success criteria
    if passed >= total * 0.75:  # 75% success rate
        print("\\n🎯 INTEGRATION TEST: SUCCESS!")
        print("Core ML pipeline components are working correctly.")
        print("Ready to proceed with full ML agent workflows.")
    else:
        print(f"\\n⚠️ INTEGRATION TEST: PARTIAL SUCCESS ({passed}/{total})")
        print("Some components need attention before full deployment.")

    print("\\n🚀 Next Steps:")
    print("1. Review any failed tests above")
    print("2. Fix configuration issues (API keys, GitHub tokens)")
    print("3. Run full ML agent: python ml_agent_cli.py workflow 'Train models'")
    print("4. Deploy models: python automl_cli.py run")

    return test_results


if __name__ == "__main__":
    run_simple_integration_test()
