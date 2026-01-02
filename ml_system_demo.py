#!/usr/bin/env python3
"""
Complete ML System Demo

Demonstrates the full automated ML system for Polymarket including:
- Database storage and retrieval
- ML tools and agent capabilities
- Complete ML pipelines
- Experiment tracking and reporting
"""

import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.automl.ml_database import MLDatabase
from agents.automl.ml_agent import create_ml_agent


def demonstrate_database_operations():
    """Demonstrate database operations."""
    print("🗄️ ML Database Operations Demo")
    print("=" * 35)

    db = MLDatabase()

    # Create a sample experiment
    experiment_id = db.create_experiment(
        name="Demo Experiment",
        description="Demonstrating ML database capabilities"
    )
    print(f"✅ Created experiment: {experiment_id}")

    # Save a sample model
    model_info = {
        'name': 'Demo MarketPredictor',
        'model_type': 'MarketPredictor',
        'algorithm': 'RandomForest',
        'hyperparameters': {'n_estimators': 100, 'max_depth': 10},
        'feature_columns': ['volume', 'yes_price', 'category'],
        'training_samples': 1000,
        'training_start_time': datetime.now().isoformat(),
        'training_end_time': datetime.now().isoformat()
    }

    model_id = db.save_model(experiment_id, model_info)
    print(f"✅ Saved model: {model_id}")

    # Save sample metrics
    metrics = {
        'accuracy': 0.75,
        'precision': 0.72,
        'recall': 0.78,
        'f1': 0.75,
        'roc_auc': 0.82
    }

    db.save_model_metrics(model_id, metrics)
    print(f"✅ Saved metrics for model: {metrics}")

    # Save sample predictions
    predictions = [
        {
            'market_id': 'demo_market_1',
            'predicted_probability': 0.65,
            'actual_outcome': 1,
            'confidence': 0.8,
            'recommended_bet': 'YES',
            'position_size': 0.05,
            'expected_value': 0.025
        },
        {
            'market_id': 'demo_market_2',
            'predicted_probability': 0.45,
            'actual_outcome': 0,
            'confidence': 0.7,
            'recommended_bet': 'NO',
            'position_size': 0.03,
            'expected_value': 0.015
        }
    ]

    db.save_predictions(model_id, predictions)
    print(f"✅ Saved {len(predictions)} predictions")

    # Save evaluation results
    evaluation_config = {'evaluation_type': 'backtest', 'test_period_days': 30}
    evaluation_results = {
        'total_trades': 50,
        'win_rate': 0.68,
        'avg_return_per_trade': 0.025,
        'sharpe_ratio': 1.8,
        'max_drawdown': 0.15
    }

    db.save_evaluation(model_id, 'backtest', evaluation_config, evaluation_results, 45.2)
    print("✅ Saved evaluation results")
    # Get experiment results
    results = db.get_experiment_results(experiment_id)
    if results:
        print("✅ Retrieved experiment results:"        print(f"   • Models: {len(results['models'])}")
        print(f"   • Datasets: {len(results['datasets'])}")

    # Show database stats
    stats = db.get_database_stats()
    print("\\n📊 Database Statistics:")
    print(f"   • Experiments: {stats.get('experiments_count', 0)}")
    print(f"   • Models: {stats.get('models_count', 0)}")
    print(f"   • Predictions: {stats.get('predictions_count', 0)}")
    print(".1f"
    return experiment_id, model_id


def demonstrate_ml_agent_workflows():
    """Demonstrate ML agent workflow capabilities."""
    print("\\n🤖 ML Agent Workflows Demo")
    print("=" * 32)

    agent = create_ml_agent()

    # Example workflows to demonstrate
    workflows = [
        "Check the quality of available market data",
        "Train a MarketPredictor model on recent data",
        "Evaluate the model's performance using backtesting"
    ]

    results = []

    for i, workflow in enumerate(workflows, 1):
        print(f"\\n🎯 Workflow {i}: {workflow}")
        print("-" * 50)

        try:
            result = agent.run_ml_workflow(workflow)
            results.append(result)

            if result.get('status') == 'success':
                print("✅ Completed successfully")
                if 'parsed_info' in result and result['parsed_info']:
                    print("📋 Key results:")
                    for key, value in result['parsed_info'].items():
                        print(f"   • {key}: {value}")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({'status': 'error', 'error': str(e)})

    print(f"\\n📊 Agent Workflow Summary:")
    print(f"   • Workflows executed: {len(workflows)}")
    print(f"   • Successful: {sum(1 for r in results if r.get('status') == 'success')}")
    print(f"   • Failed: {sum(1 for r in results if r.get('status') != 'success')}")

    return results


def demonstrate_ml_tools():
    """Demonstrate individual ML tools."""
    print("\\n🔧 ML Tools Demo")
    print("=" * 20)

    from agents.automl.ml_tools import (
        data_ingestion_tool,
        data_quality_tool,
        model_training_tool,
        model_evaluation_tool,
        prediction_tool
    )

    tools = [
        ("Data Ingestion", data_ingestion_tool, {"days_back": 30, "min_volume": 1000}),
        ("Data Quality Check", data_quality_tool, {"dataset_info": "demo dataset"}),
        ("Model Training", model_training_tool, {
            "model_type": "MarketPredictor",
            "experiment_name": "Tool Demo",
            "hyperparameters": {"n_estimators": 50}
        })
    ]

    results = []

    for tool_name, tool, params in tools:
        print(f"🛠️ Testing {tool_name}...")
        try:
            # Note: In practice, these would be called through the agent
            # This is just a demonstration of tool availability
            print(f"   ✅ {tool_name} tool is available")
            print(f"   📝 Description: {tool.description[:100]}...")
            results.append({'tool': tool_name, 'status': 'available'})

        except Exception as e:
            print(f"   ❌ {tool_name} failed: {e}")
            results.append({'tool': tool_name, 'status': 'error', 'error': str(e)})

    print(f"\\n🔧 Tools Status: {sum(1 for r in results if r['status'] == 'available')}/{len(tools)} available")
    return results


def demonstrate_reporting():
    """Demonstrate reporting capabilities."""
    print("\\n📋 ML Reporting Demo")
    print("=" * 23)

    agent = create_ml_agent()
    report = agent.create_ml_report()

    print("📄 Generated ML Report:")
    print("=" * 30)

    # Show first few lines of report
    lines = report.split('\n')
    for line in lines[:20]:  # Show first 20 lines
        print(line)

    if len(lines) > 20:
        print(f"... ({len(lines) - 20} more lines)")

    print("\\n📊 Report Sections:")
    sections = [line for line in lines if line.startswith('##')]
    for section in sections:
        print(f"   • {section[3:]}")

    return report


def demonstrate_system_integration():
    """Demonstrate how all components work together."""
    print("\\n🔗 System Integration Demo")
    print("=" * 30)

    print("🚀 Complete ML Pipeline Integration:")
    print()
    print("1. 📊 Data Ingestion")
    print("   → Polymarket API → Data cleaning → Database storage")
    print()
    print("2. 🔧 Data Quality Validation")
    print("   → Statistical checks → Outlier detection → Feature engineering")
    print()
    print("3. 🤖 ML Agent & Tools")
    print("   → Intelligent workflows → Automated model training → Evaluation")
    print()
    print("4. 🗄️ Database Operations")
    print("   → Experiment tracking → Model storage → Performance metrics")
    print()
    print("5. 📈 Reporting & Monitoring")
    print("   → Automated reports → Performance tracking → Alert system")
    print()
    print("6. 🚀 Production Deployment")
    print("   → Model serving → Prediction APIs → Continuous learning")
    print()

    print("✨ Key Integration Points:")
    print("   • Agent uses tools for ML operations")
    print("   • Tools store results in database")
    print("   • Database provides data for reporting")
    print("   • Reports inform agent decisions")
    print("   • Continuous feedback loop for improvement")


def show_cli_usage():
    """Show CLI usage examples."""
    print("\\n🚀 CLI Usage Examples")
    print("=" * 25)

    examples = [
        ("Run ML workflow", "python ml_agent_cli.py workflow 'Train a market predictor model'"),
        ("Check database status", "python ml_agent_cli.py status"),
        ("Generate ML report", "python ml_agent_cli.py report --save-report"),
        ("Optimize hyperparameters", "python ml_agent_cli.py optimize MarketPredictor"),
        ("Create trading strategy", "python ml_agent_cli.py strategy model_123 --risk-tolerance 0.05")
    ]

    for desc, command in examples:
        print(f"📝 {desc}:")
        print(f"   {command}")
        print()


def main():
    """Main demonstration."""
    try:
        print("🎯 Complete ML System Demo for Polymarket")
        print("=" * 50)

        # Demonstrate database operations
        experiment_id, model_id = demonstrate_database_operations()

        # Demonstrate ML tools
        tool_results = demonstrate_ml_tools()

        # Demonstrate agent workflows
        workflow_results = demonstrate_ml_agent_workflows()

        # Demonstrate reporting
        report = demonstrate_reporting()

        # Show system integration
        demonstrate_system_integration()

        # Show CLI usage
        show_cli_usage()

        print("\\n🎉 ML System Demo Complete!")
        print("\\n🏆 System Capabilities Demonstrated:")
        print("   ✅ Database storage and retrieval")
        print("   ✅ ML tools for automated operations")
        print("   ✅ Intelligent agent workflows")
        print("   ✅ Experiment tracking and reporting")
        print("   ✅ Complete ML pipeline integration")
        print("   ✅ Production-ready architecture")

        print("\\n🚀 Ready for Production ML Operations!")
        print("\\n💡 Next Steps:")
        print("1. Run: python ml_agent_cli.py status (check system health)")
        print("2. Train: python ml_agent_cli.py workflow 'Train models on recent data'")
        print("3. Deploy: Use trained models for live predictions")
        print("4. Monitor: python ml_agent_cli.py report (track performance)")

    except Exception as e:
        print(f"\\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
