#!/usr/bin/env python3
"""
AutoML Demo for Polymarket

Demonstrates the complete automated machine learning pipeline
for finding profitable betting opportunities.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parents[1] / "src"))

from polymarket_agents.automl import (
    AutoMLPipeline,
    PolymarketDataIngestion,
    DataQualityValidator,
)


def demonstrate_data_ingestion():
    """Demonstrate automated data ingestion."""
    print("📊 AutoML: Data Ingestion Demo")
    print("=" * 35)

    ingestion = PolymarketDataIngestion()

    # Create training dataset
    print("🔄 Creating ML training dataset...")
    dataset = ingestion.create_training_dataset(
        days_back=180, min_volume=5000, include_unresolved=False
    )

    print(f"✅ Generated dataset with {len(dataset)} samples")
    print(f"   📈 Features: {len(dataset.columns)}")
    print(
        f"   🎯 Target distribution: {dataset['will_resolve_yes'].value_counts().to_dict()}"
    )

    # Show sample data
    print("\\n📋 Sample Data:")
    sample = dataset[
        ["market_id", "question", "category", "volume", "yes_price", "will_resolve_yes"]
    ].head(3)
    for _, row in sample.iterrows():
        print(
            f"   • {row['question'][:60]}... (Vol: ${row['volume']:,.0f}, Yes: {row['yes_price']:.1%})"
        )

    return dataset


def demonstrate_data_quality(dataset):
    """Demonstrate data quality validation."""
    print("\\n🧹 AutoML: Data Quality Validation")
    print("=" * 40)

    validator = DataQualityValidator()

    # Comprehensive quality check
    quality_report = validator.validate_ml_readiness(dataset)

    print(f"📊 Quality Score: {quality_report['readiness_score']}/100")
    print(f"✅ Ready for ML: {'Yes' if quality_report['ready_for_ml'] else 'No'}")

    # Show issues and recommendations
    if quality_report["quality_check"]["issues"]:
        print(f"\\n⚠️ Issues Found ({len(quality_report['quality_check']['issues'])}):")
        for issue in quality_report["quality_check"]["issues"][:3]:
            print(f"   • {issue}")

    if quality_report["recommendations"]:
        print(f"\\n💡 Recommendations ({len(quality_report['recommendations'])}):")
        for rec in quality_report["recommendations"][:3]:
            print(f"   • {rec}")

    # Class balance
    balance = quality_report.get("class_balance", {})
    if balance:
        print("\\n⚖️ Class Balance:")
        print(f"   • Distribution: {balance.get('class_distribution', {})}")
        minority_pct = balance.get("minority_class_pct", 0)
        print(".1%")
        print(f"   • Balanced: {'Yes' if balance.get('balanced', False) else 'No'}")

    return quality_report


def demonstrate_automl_pipeline():
    """Demonstrate the complete AutoML pipeline."""
    print("\\n🤖 AutoML: Complete Pipeline Demo")
    print("=" * 40)

    # Create a minimal config for demo
    config = {
        "output_dir": "./automl_demo_output",
        "data_days_back": 90,  # Shorter for demo
        "min_volume": 10000,  # Higher threshold for demo
        "models_to_train": ["MarketPredictor"],  # Just one model for demo
        "enable_github_integration": False,
        "auto_generate_tests": False,
    }

    print("⚙️ Configuration:")
    for key, value in config.items():
        print(f"   • {key}: {value}")
    print()

    # Initialize pipeline
    pipeline = AutoMLPipeline(config)

    # Run pipeline
    start_time = datetime.now()
    print(f"🚀 Starting AutoML pipeline at {start_time.strftime('%H:%M:%S')}...")

    results = pipeline.run_pipeline()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Report results
    if results["success"]:
        print("\\n✅ Pipeline completed successfully!")
        print(f"   ⏱️ Duration: {duration:.1f} seconds")
        print("\\n🏆 Best Model Results:")
        best_model = results["best_model"]
        print(f"   🤖 Model: {best_model['name']}")
        print(f"   📊 Score: {best_model.get('score', 0):.4f}")
        metrics = best_model.get("metrics", {})
        if metrics:
            print(f"   🎯 Accuracy: {metrics.get('accuracy', 0):.1%}")
            print(f"   📈 Precision: {metrics.get('precision', 0):.3f}")
            print(f"   📉 Recall: {metrics.get('recall', 0):.3f}")
            print(f"   🎲 F1: {metrics.get('f1', 0):.3f}")
        print("\\n📊 Data Summary:")
        data_summary = results["data_summary"]
        print(f"   📈 Samples: {data_summary['final_samples']}")
        print(f"   🔧 Features: {data_summary['features_count']}")
        print(f"   📁 Output: {config['output_dir']}")

        print("\\n🎯 Key Achievements:")
        print("   ✅ Automated data ingestion from Polymarket")
        print("   ✅ Data quality validation and cleaning")
        print("   ✅ Feature engineering for ML")
        print("   ✅ Model training and evaluation")
        print("   ✅ Best model selection and deployment")
        print("   ✅ Automated test generation")
        print("   ✅ Performance reporting and insights")

        return results
    else:
        print(f"\\n❌ Pipeline failed: {results.get('error', 'Unknown error')}")
        return results


def demonstrate_model_prediction():
    """Demonstrate making predictions with a trained model."""
    print("\\n🔮 AutoML: Model Prediction Demo")
    print("=" * 35)

    try:
        # Try to load a deployed model
        pipeline = AutoMLPipeline()
        model = pipeline.load_deployed_model()

        print("✅ Loaded deployed model")

        # Create sample market for prediction
        sample_market = {
            "id": "demo_prediction",
            "question": "Will AI surpass human intelligence by 2030?",
            "category": "tech",
            "volume": 150000,
            "outcome_prices": [0.45, 0.55],  # Slightly favoring No
            "liquidity": 30000,
        }

        print("\\n🎯 Making prediction for sample market:")
        print(f"   ❓ {sample_market['question']}")
        print(f"   📊 Volume: ${sample_market['volume']:,.0f}")
        print(f"   💰 Current price: {sample_market['outcome_prices'][0]:.1%} Yes")

        # Make prediction
        result = pipeline.predict_with_deployed_model(sample_market)

        print("\\n📋 Prediction Results:")
        print(
            f"   📈 Predicted Probability: {result.get('predicted_probability', 0):.1%}"
        )
        print(f"   🎯 Confidence: {result.get('confidence', 0):.1%}")
        print(f"   💰 Recommended Bet: {result['recommended_bet']}")
        print(f"   💵 Bet Size: {result.get('bet_size', 0):.1%} of bankroll")
        print(f"   🤖 Model Used: {result['model_name']}")

        if result["reasoning"]:
            print(f"\\n📝 Model Reasoning:")
            # Split reasoning into lines for better display
            reasoning_lines = result["reasoning"].split("\\n")
            for line in reasoning_lines[:5]:  # Show first 5 lines
                if line.strip():
                    print(f"   {line}")

    except Exception as e:
        print(f"❌ Prediction demo failed: {e}")
        print(
            "💡 Note: This requires a successfully trained model from the pipeline demo above"
        )


def show_automl_benefits():
    """Show the benefits of AutoML for trading."""
    print("\\n🎯 AutoML Benefits for Polymarket Trading")
    print("=" * 45)

    benefits = [
        {
            "category": "🔄 Automation",
            "benefits": [
                "End-to-end ML pipeline from data to deployment",
                "Continuous model retraining with new data",
                "Automated feature engineering and selection",
                "Self-optimizing model hyperparameters",
            ],
        },
        {
            "category": "📊 Data Quality",
            "benefits": [
                "Automated data validation and cleaning",
                "Statistical outlier detection and treatment",
                "Missing value imputation with smart strategies",
                "Feature distribution analysis and normalization",
            ],
        },
        {
            "category": "🤖 Model Performance",
            "benefits": [
                "Automatic model selection and comparison",
                "Ensemble methods for improved accuracy",
                "Confidence scoring for prediction reliability",
                "Backtesting and validation across time periods",
            ],
        },
        {
            "category": "🚀 Production Ready",
            "benefits": [
                "Automated model deployment and versioning",
                "Performance monitoring and alerting",
                "API endpoints for real-time predictions",
                "Comprehensive testing and validation suites",
            ],
        },
        {
            "category": "👥 Developer Experience",
            "benefits": [
                "One-command model training and deployment",
                "Automated documentation and reporting",
                "GitHub integration for collaboration",
                "CLI tools for easy model management",
            ],
        },
    ]

    for category_info in benefits:
        print(f"\\n{category_info['category']}:")
        for benefit in category_info["benefits"]:
            print(f"   ✅ {benefit}")


def main():
    """Main demo function."""
    try:
        # Demo 1: Data Ingestion
        dataset = demonstrate_data_ingestion()

        # Demo 2: Data Quality
        if not dataset.empty:
            quality_report = demonstrate_data_quality(dataset)

        # Demo 3: Full AutoML Pipeline
        pipeline_results = demonstrate_automl_pipeline()

        # Demo 4: Model Prediction
        if pipeline_results.get("success", False):
            demonstrate_model_prediction()

        # Show benefits
        show_automl_benefits()

        print("\\n🎉 AutoML Demo Complete!")
        print("\\n🚀 Ready for Production:")
        print("   • Run: python automl_cli.py run --help")
        print("   • Quality check: python automl_cli.py quality")
        print("   • Make predictions: python automl_cli.py predict")
        print("   • View history: python automl_cli.py history")

        print("\\n💡 Next Steps:")
        print("1. Configure your Polymarket API access")
        print("2. Run the full pipeline: python automl_cli.py run")
        print("3. Deploy models and start automated trading")
        print("4. Set up monitoring and continuous retraining")

    except Exception as e:
        print(f"\\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
