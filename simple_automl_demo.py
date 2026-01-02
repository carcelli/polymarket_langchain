#!/usr/bin/env python3
"""
Simple AutoML Demo for Polymarket

Shows the core AutoML functionality without complex formatting.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.automl import PolymarketDataIngestion, DataQualityValidator


def demo_data_ingestion():
    """Demo data ingestion."""
    print("📊 AutoML: Data Ingestion")
    print("=" * 30)

    ingestion = PolymarketDataIngestion()

    print("🔄 Creating training dataset...")
    dataset = ingestion.create_training_dataset(days_back=90, min_volume=1000)

    print(f"✅ Dataset created: {len(dataset)} samples, {len(dataset.columns)} features")

    if not dataset.empty:
        print(f"🎯 Target distribution: {dataset['will_resolve_yes'].value_counts().to_dict()}")

        # Show sample
        sample = dataset.iloc[0]
        print(f"📋 Sample market: {sample['question'][:50]}...")
        print(".1%")

    return dataset


def demo_data_quality(dataset):
    """Demo data quality validation."""
    print("\n🧹 AutoML: Data Quality")
    print("=" * 30)

    validator = DataQualityValidator()
    quality_report = validator.validate_ml_readiness(dataset)

    print(f"📊 Quality Score: {quality_report['readiness_score']}/100")
    print(f"✅ Ready for ML: {'Yes' if quality_report['ready_for_ml'] else 'No'}")

    # Show issues
    issues = quality_report['quality_check']['issues']
    if issues:
        print(f"⚠️ Found {len(issues)} issues")
        for issue in issues[:2]:
            print(f"   • {issue}")

    # Class balance
    balance = quality_report.get('class_balance', {})
    if balance:
        print("⚖️ Class Balance:")
        print(f"   • Distribution: {balance.get('class_distribution', {})}")
        minority_pct = balance.get('minority_class_pct', 0)
        print(f"   • Minority class: {minority_pct:.1f}%")
        print(f"   • Balanced: {'Yes' if balance.get('balanced', False) else 'No'}")

    return quality_report


def demo_automl_components():
    """Show AutoML component integration."""
    print("\n🤖 AutoML: Component Integration")
    print("=" * 35)

    print("✅ Available AutoML Components:")
    print("   📊 PolymarketDataIngestion - Automated data collection")
    print("   🧹 DataQualityValidator - Data validation & cleaning")
    print("   🔧 Feature Engineering - ML-ready feature creation")
    print("   🤖 Model Training - Automated model training & selection")
    print("   📋 Test Generation - Automated test suite creation")
    print("   🚀 Model Deployment - Production model serving")

    print("\n🔄 AutoML Pipeline Steps:")
    print("   1. Data Ingestion from Polymarket API")
    print("   2. Quality validation and cleaning")
    print("   3. Feature engineering and preprocessing")
    print("   4. Model training and evaluation")
    print("   5. Best model selection and validation")
    print("   6. Automated test generation")
    print("   7. Model deployment and monitoring")

    print("\n🎯 Key Benefits:")
    print("   • End-to-end automation from data to deployment")
    print("   • Continuous model improvement with new data")
    print("   • Automated quality assurance and testing")
    print("   • Production-ready model serving")
    print("   • Comprehensive performance monitoring")


def show_cli_usage():
    """Show CLI usage examples."""
    print("\n🚀 AutoML CLI Usage")
    print("=" * 25)

    print("Run full AutoML pipeline:")
    print("  python automl_cli.py run --days-back 365 --models MarketPredictor EdgeDetector")

    print("\nCheck data quality:")
    print("  python automl_cli.py quality --days-back 180")

    print("\nMake predictions:")
    print("  python automl_cli.py predict --question 'Will BTC reach $100k?' --price 0.6")

    print("\nView pipeline history:")
    print("  python automl_cli.py history")


def main():
    """Main demo."""
    try:
        print("🤖 Polymarket AutoML Demo")
        print("=" * 30)

        # Demo data ingestion
        dataset = demo_data_ingestion()

        # Demo data quality (if we have data)
        if not dataset.empty:
            quality_report = demo_data_quality(dataset)

        # Show components
        demo_automl_components()

        # Show CLI usage
        show_cli_usage()

        print("\n🎉 AutoML Demo Complete!")
        print("\n💡 Your system now has:")
        print("   ✅ Automated data ingestion from Polymarket")
        print("   ✅ Data quality validation and cleaning")
        print("   ✅ ML-ready feature engineering")
        print("   ✅ Automated model training pipelines")
        print("   ✅ Production model deployment")
        print("   ✅ Continuous integration and testing")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
