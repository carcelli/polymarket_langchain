#!/usr/bin/env python3
"""
ML Agent CLI - Command Line Interface for ML Operations

Provides easy access to ML agent capabilities and database management.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parents[2] / "src"))

from polymarket_agents.automl.ml_agent import create_ml_agent
from polymarket_agents.automl.ml_database import MLDatabase


def run_ml_workflow(args):
    """Run an ML workflow using the ML Agent."""
    print("🤖 ML Agent Workflow")
    print("=" * 30)

    agent = create_ml_agent()

    # Run the workflow
    start_time = datetime.now()
    print(f"🎯 Executing: {args.task}")
    print(f"⏰ Started at: {start_time.strftime('%H:%M:%S')}")

    result = agent.run_ml_workflow(args.task)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Display results
    if result.get("status") == "success":
        print("\\n✅ Workflow completed successfully!")
        print(f"   ⏱️ Duration: {duration:.1f} seconds")
        if "parsed_info" in result:
            print("\\n📋 Key Results:")
            for key, value in result["parsed_info"].items():
                print(f"   • {key}: {value}")
    else:
        print(f"\\n❌ Workflow failed: {result.get('error', 'Unknown error')}")

    # Save results if requested
    if args.save_results:
        results_file = (
            f"ml_workflow_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"💾 Results saved to: {results_file}")

    return result


def show_database_status(args):
    """Show database status and recent activity."""
    print("🗄️ ML Database Status")
    print("=" * 25)

    db = MLDatabase()

    try:
        stats = db.get_database_stats()

        print("📊 Database Statistics:")
        print(f"   📁 Database: {db.db_path}")
        print(f"   📏 Size: {stats.get('database_size_bytes', 0) / 1024:.1f} KB")
        print(f"   🧪 Experiments: {stats.get('experiments_count', 0)}")
        print(f"   🤖 Models: {stats.get('models_count', 0)}")
        print(f"   🎯 Predictions: {stats.get('predictions_count', 0)}")
        print(f"   ⚡ Evaluations: {stats.get('evaluations_count', 0)}")
        print(
            f"   📈 Recent Activity (7 days): {stats.get('experiments_last_7_days', 0)} experiments"
        )

        # Show recent experiments
        print("\\n🕐 Recent Experiments:")
        # This would query recent experiments - simplified for now
        print("   (Recent experiments would be listed here)")

        # Show active alerts
        alerts = db.get_active_alerts()
        if alerts:
            print(f"\\n🚨 Active Alerts ({len(alerts)}):")
            for alert in alerts[:5]:
                print(f"   {alert['severity'].upper()}: {alert['message']}")

        # Show best models
        best_models = db.get_best_models(limit=3)
        if not best_models.empty:
            print("\\n🏆 Best Performing Models:")
            for _, model in best_models.iterrows():
                f1_score = model.get("f1_score", 0)
                print(f"   • {model.get('model_name', 'Unknown')}: F1={f1_score:.3f}")
    except Exception as e:
        print(f"❌ Database error: {e}")


def generate_ml_report(args):
    """Generate comprehensive ML report."""
    print("📋 ML Report Generation")
    print("=" * 25)

    agent = create_ml_agent()
    report = agent.create_ml_report()

    print("📄 ML Report:")
    print("=" * 50)
    print(report)

    # Save report if requested
    if args.save_report:
        report_file = f"ml_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, "w") as f:
            f.write(report)
        print(f"\\n💾 Report saved to: {report_file}")


def optimize_hyperparameters(args):
    """Optimize model hyperparameters."""
    print("🎛️ Hyperparameter Optimization")
    print("=" * 35)

    agent = create_ml_agent()

    # Default parameter grids
    param_grids = {
        "MarketPredictor": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
        },
        "EdgeDetector": {
            "hidden_layers": [[32], [64, 32], [128, 64, 32]],
            "learning_rate": [0.001, 0.01, 0.1],
            "dropout_rate": [0.1, 0.2, 0.3],
        },
    }

    param_grid = param_grids.get(args.model_type, {})

    print(f"🎯 Optimizing {args.model_type} hyperparameters")
    print(f"📊 Parameter combinations: {len(param_grid)} parameters")

    result = agent.optimize_model_hyperparameters(args.model_type, param_grid)

    if result.get("status") == "success":
        print("✅ Optimization completed!")
        if "best_params" in result.get("parsed_info", {}):
            print(f"🏆 Best parameters: {result['parsed_info']['best_params']}")
    else:
        print(f"❌ Optimization failed: {result.get('error', 'Unknown error')}")


def check_model_drift(args):
    """Check for model drift."""
    print("📈 Model Drift Detection")
    print("=" * 28)

    agent = create_ml_agent()

    print(f"🔍 Checking drift for model: {args.model_id}")

    result = agent.validate_model_drift(args.model_id)

    if result.get("status") == "success":
        print("✅ Drift check completed!")
        # Display drift metrics
        print("📊 Drift Analysis:")
        print("   (Detailed drift metrics would be displayed here)")
    else:
        print(f"❌ Drift check failed: {result.get('error', 'Unknown error')}")


def retrain_model(args):
    """Retrain a model with new data."""
    print("🔄 Model Retraining")
    print("=" * 22)

    agent = create_ml_agent()

    print(f"🔄 Retraining model: {args.model_id}")
    print(f"📅 New data period: {args.new_data_days} days")

    result = agent.retrain_model(args.model_id, args.new_data_days)

    if result.get("status") == "success":
        print("✅ Retraining completed!")
        print("📊 Performance comparison:")
        print("   (Old vs new model metrics would be displayed here)")
    else:
        print(f"❌ Retraining failed: {result.get('error', 'Unknown error')}")


def create_trading_strategy(args):
    """Create a trading strategy from ML model."""
    print("📈 Trading Strategy Generation")
    print("=" * 35)

    agent = create_ml_agent()

    print(f"🎯 Creating strategy from model: {args.model_id}")
    print(f"⚠️ Risk tolerance: {args.risk_tolerance}")

    result = agent.generate_trading_strategy(args.model_id, args.risk_tolerance)

    if result.get("status") == "success":
        print("✅ Trading strategy created!")
        print("📋 Strategy Specification:")
        print("   (Strategy rules and parameters would be displayed here)")
    else:
        print(f"❌ Strategy creation failed: {result.get('error', 'Unknown error')}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ML Agent CLI - Automated Machine Learning for Polymarket",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run ML workflow
  python ml_agent_cli.py workflow "Train a market predictor model"

  # Check database status
  python ml_agent_cli.py status

  # Generate ML report
  python ml_agent_cli.py report --save-report

  # Optimize hyperparameters
  python ml_agent_cli.py optimize MarketPredictor

  # Check model drift
  python ml_agent_cli.py drift model_20241201_120000_marketpredictor

  # Retrain model
  python ml_agent_cli.py retrain model_20241201_120000_marketpredictor --new-data-days 30

  # Create trading strategy
  python ml_agent_cli.py strategy model_20241201_120000_marketpredictor --risk-tolerance 0.05
        """,
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Workflow command
    workflow_parser = subparsers.add_parser("workflow", help="Run ML workflow")
    workflow_parser.add_argument("task", help="ML task description")
    workflow_parser.add_argument(
        "--save-results", action="store_true", help="Save results to JSON file"
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Show database status")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate ML report")
    report_parser.add_argument(
        "--save-report", action="store_true", help="Save report to markdown file"
    )

    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize hyperparameters")
    optimize_parser.add_argument(
        "model_type",
        choices=["MarketPredictor", "EdgeDetector"],
        help="Model type to optimize",
    )

    # Drift command
    drift_parser = subparsers.add_parser("drift", help="Check model drift")
    drift_parser.add_argument("model_id", help="Model ID to check for drift")

    # Retrain command
    retrain_parser = subparsers.add_parser(
        "retrain", help="Retrain model with new data"
    )
    retrain_parser.add_argument("model_id", help="Model ID to retrain")
    retrain_parser.add_argument(
        "--new-data-days", type=int, default=30, help="Days of new data to include"
    )

    # Strategy command
    strategy_parser = subparsers.add_parser("strategy", help="Create trading strategy")
    strategy_parser.add_argument("model_id", help="Model ID to base strategy on")
    strategy_parser.add_argument(
        "--risk-tolerance",
        type=float,
        default=0.1,
        help="Risk tolerance for position sizing",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "workflow":
            run_ml_workflow(args)
        elif args.command == "status":
            show_database_status(args)
        elif args.command == "report":
            generate_ml_report(args)
        elif args.command == "optimize":
            optimize_hyperparameters(args)
        elif args.command == "drift":
            check_model_drift(args)
        elif args.command == "retrain":
            retrain_model(args)
        elif args.command == "strategy":
            create_trading_strategy(args)

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
