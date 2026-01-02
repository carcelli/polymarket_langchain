#!/usr/bin/env python3
"""
AutoML CLI for Polymarket

Command-line interface for automated machine learning on Polymarket data.
Provides complete end-to-end ML pipeline from data ingestion to model deployment.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.automl import AutoMLPipeline, PolymarketDataIngestion, DataQualityValidator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_automl_pipeline(args):
    """Run the complete AutoML pipeline."""
    print("🤖 Polymarket AutoML Pipeline")
    print("=" * 40)

    # Configuration
    config = {
        'output_dir': args.output_dir,
        'data_days_back': args.days_back,
        'min_volume': args.min_volume,
        'test_size': args.test_size,
        'models_to_train': args.models,
        'enable_github_integration': args.github,
        'auto_generate_tests': args.generate_tests
    }

    print(f"Configuration:")
    print(f"  📊 Data: {args.days_back} days back, min volume ${args.min_volume:,.0f}")
    print(f"  🤖 Models: {', '.join(args.models)}")
    print(f"  📁 Output: {args.output_dir}")
    print(f"  🔗 GitHub: {'Enabled' if args.github else 'Disabled'}")
    print()

    # Initialize pipeline
    pipeline = AutoMLPipeline(config)

    # Run pipeline
    start_time = datetime.now()
    print(f"🚀 Starting pipeline at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = pipeline.run_pipeline()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Report results
    if results['success']:
        print("\\n✅ Pipeline completed successfully!")
        print(".1f"        print("\\n📊 Results Summary:")
        print(f"   📈 Best Model: {results['best_model']['name']}")
        print(".4f"        print(f"   📊 Data Samples: {results['data_summary']['final_samples']}")
        print(f"   🔧 Features: {results['data_summary']['features_count']}")
        print(f"   📁 Output Directory: {args.output_dir}")

        if args.github and 'tests' in results.get('reports', {}):
            print("\\n🧪 Automated Tests Generated:")
            test_files = results['reports']['tests'].get('files', {})
            for filename in test_files.keys():
                print(f"   • {filename}")

        print("\\n🎯 Key Insights:")
        best_metrics = results['best_model'].get('metrics', {})
        if best_metrics:
            print(".1%"            print(".3f"            print(".3f"    else:
        print("\\n❌ Pipeline failed!")
        print(f"Error: {results.get('error', 'Unknown error')}")

    return results


def check_data_quality(args):
    """Check data quality without running full pipeline."""
    print("🧹 Data Quality Check")
    print("=" * 25)

    # Load data
    ingestion = PolymarketDataIngestion()
    data = ingestion.create_training_dataset(days_back=args.days_back, min_volume=args.min_volume)

    if data.empty:
        print("❌ No data available")
        return

    # Quality check
    validator = DataQualityValidator()
    quality_report = validator.validate_ml_readiness(data)

    print("\\n📊 Data Quality Report:")
    print(f"   📈 Readiness Score: {quality_report['readiness_score']}/100")
    print(f"   📋 Ready for ML: {'✅ Yes' if quality_report['ready_for_ml'] else '❌ No'}")
    print(f"   📏 Dataset Shape: {quality_report['dataset_info']['shape']}")

    if quality_report['recommendations']:
        print("\\n💡 Recommendations:")
        for rec in quality_report['recommendations'][:5]:
            print(f"   • {rec}")

    # Class balance
    balance = quality_report.get('class_balance', {})
    if balance:
        print("\\n⚖️ Class Balance:")
        dist = balance.get('class_distribution', {})
        for cls, count in dist.items():
            pct = count / balance.get('total_samples', 1) * 100
            print(".1f"    return quality_report


def predict_with_model(args):
    """Make predictions using a deployed model."""
    print("🔮 Model Prediction")
    print("=" * 20)

    try:
        pipeline = AutoMLPipeline()
        model = pipeline.load_deployed_model(args.model_name)

        # Create sample market data (in practice, this would come from API)
        sample_market = {
            'id': 'prediction_test',
            'question': args.question or 'Will this prediction be accurate?',
            'category': args.category or 'politics',
            'volume': args.volume or 50000,
            'outcome_prices': [args.price, 1 - args.price],
            'liquidity': args.liquidity or 10000
        }

        result = pipeline.predict_with_deployed_model(sample_market)

        print("\\n🎯 Prediction Results:")
        print(f"   ❓ Question: {sample_market['question']}")
        print(".1%"        print(".1%"        print(f"   💰 Recommended Bet: {result['recommended_bet']}")
        print(".1%"        print(f"   🤖 Model: {result['model_name']}")

        if result['reasoning']:
            print(f"   📝 Reasoning: {result['reasoning'][:200]}...")

    except Exception as e:
        print(f"❌ Prediction failed: {e}")


def show_pipeline_history(args):
    """Show history of pipeline runs."""
    print("📚 Pipeline History")
    print("=" * 20)

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}")
        return

    # Find pipeline result files
    result_files = list(output_dir.glob("automl_pipeline_results_*.json"))

    if not result_files:
        print("❌ No pipeline results found")
        return

    # Sort by modification time (newest first)
    result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    print(f"Found {len(result_files)} pipeline runs:")
    print()

    for i, result_file in enumerate(result_files[:10]):  # Show last 10
        try:
            with open(result_file, 'r') as f:
                results = json.load(f)

            success = results.get('success', False)
            status = "✅ Success" if success else "❌ Failed"

            start_time = results.get('pipeline_metadata', {}).get('start_time', 'Unknown')
            duration = results.get('pipeline_metadata', {}).get('duration_seconds', 0)

            if success:
                best_model = results.get('best_model', {}).get('name', 'Unknown')
                score = results.get('best_model', {}).get('score', 0)
                samples = results.get('data_summary', {}).get('final_samples', 0)

                print(f"{i+1}. {result_file.name}")
                print(f"   {status} - {best_model} (score: {score:.4f})")
                print(f"   📊 {samples} samples, {duration:.1f}s")
                print(f"   🕐 {start_time}")
            else:
                error = results.get('error', 'Unknown error')
                print(f"{i+1}. {result_file.name}")
                print(f"   {status} - {error[:50]}...")
                print(f"   🕐 {start_time}")

            print()

        except Exception as e:
            print(f"❌ Error reading {result_file.name}: {e}")
            print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AutoML CLI for Polymarket - Automated Machine Learning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full AutoML pipeline
  python automl_cli.py run --days-back 365 --models MarketPredictor EdgeDetector

  # Check data quality
  python automl_cli.py quality --days-back 180

  # Make predictions
  python automl_cli.py predict --question "Will BTC reach $100k?" --price 0.6

  # Show pipeline history
  python automl_cli.py history
        """
    )

    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Run pipeline command
    run_parser = subparsers.add_parser('run', help='Run complete AutoML pipeline')
    run_parser.add_argument('--days-back', type=int, default=365,
                           help='Days of historical data to use')
    run_parser.add_argument('--min-volume', type=float, default=1000,
                           help='Minimum market volume threshold')
    run_parser.add_argument('--test-size', type=float, default=0.2,
                           help='Test set size (0.0-1.0)')
    run_parser.add_argument('--models', nargs='+',
                           default=['MarketPredictor', 'EdgeDetector'],
                           help='Models to train')
    run_parser.add_argument('--output-dir', type=str, default='./automl_output',
                           help='Output directory for results')
    run_parser.add_argument('--github', action='store_true',
                           help='Enable GitHub integration for test commits')
    run_parser.add_argument('--generate-tests', action='store_true', default=True,
                           help='Generate automated tests')

    # Quality check command
    quality_parser = subparsers.add_parser('quality', help='Check data quality')
    quality_parser.add_argument('--days-back', type=int, default=180,
                               help='Days of data to check')
    quality_parser.add_argument('--min-volume', type=float, default=1000,
                               help='Minimum volume threshold')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions with deployed model')
    predict_parser.add_argument('--question', type=str,
                               help='Market question')
    predict_parser.add_argument('--category', type=str, default='politics',
                               help='Market category')
    predict_parser.add_argument('--price', type=float, default=0.5,
                               help='Current market price (0.0-1.0)')
    predict_parser.add_argument('--volume', type=int, default=50000,
                               help='Market volume')
    predict_parser.add_argument('--liquidity', type=int, default=10000,
                               help='Market liquidity')
    predict_parser.add_argument('--model-name', type=str,
                               help='Specific model name to use')

    # History command
    history_parser = subparsers.add_parser('history', help='Show pipeline run history')
    history_parser.add_argument('--output-dir', type=str, default='./automl_output',
                               help='Output directory to check')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == 'run':
            run_automl_pipeline(args)
        elif args.command == 'quality':
            check_data_quality(args)
        elif args.command == 'predict':
            predict_with_model(args)
        elif args.command == 'history':
            show_pipeline_history(args)

    except KeyboardInterrupt:
        print("\\n\\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
