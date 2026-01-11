#!/usr/bin/env python3
"""
Workflow CLI - Run Automated ML Workflows

Command-line interface for executing complete automated ML workflows
from data collection through model deployment and GitHub integration.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from automated_ml_workflow import run_automated_ml_workflow
from agents.automl.ml_database import MLDatabase


def run_workflow(args):
    """Run the complete automated ML workflow."""
    print("🚀 Automated ML Workflow")
    print("=" * 30)

    # Build configuration
    config = {
        'days_back': args.days_back,
        'min_volume': args.min_volume,
        'models': args.models if args.models else ['MarketPredictor', 'EdgeDetector'],
        'output_dir': args.output_dir
    }

    print("Configuration:")
    print(f"  📊 Data: {args.days_back} days back")
    print(f"  💰 Min Volume: ${args.min_volume:,}")
    print(f"  🤖 Models: {', '.join(config['models'])}")
    print(f"  📁 Output: {args.output_dir}")
    print()

    # Confirm execution
    if not args.yes:
        confirm = input("This will run a complete ML pipeline. Continue? (y/N): ")
        if confirm.lower() not in ['y', 'yes']:
            print("Workflow cancelled.")
            return

    # Run workflow
    start_time = datetime.now()
    print(f"⏰ Started at: {start_time.strftime('%H:%M:%S')}")

    try:
        results = run_automated_ml_workflow(config)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Report results
        if results.get('success'):
            print("\\n✅ WORKFLOW COMPLETED SUCCESSFULLY")
            print("=" * 40)

            summary = results.get('summary', {})

            print("📊 Results Summary:")
            print(f"   Workflow ID: {results['workflow_id']}")
            print(".1f"            print(f"   Data Samples: {summary.get('data_samples', 0)}")
            print(f"   Models Trained: {summary.get('models_trained', 0)}")

            best_score = summary.get('best_model_score')
            if best_score:
                print(".3f"            print(f"   GitHub Commits: {summary.get('github_commits', 0)}")

            # Save detailed results
            if args.save_results:
                results_file = f"workflow_results_{results['workflow_id']}.json"
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\\n💾 Detailed results saved to: {results_file}")

        else:
            print("\\n❌ WORKFLOW FAILED")
            print("=" * 20)
            print(f"Error: {results.get('error', 'Unknown error')}")
            print(f"Failed at phase: {results.get('failed_at_phase', 'unknown')}")

    except KeyboardInterrupt:
        print("\\n\\n⚠️ Workflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n❌ Workflow error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


def show_workflow_history(args):
    """Show history of completed workflows."""
    print("📚 Workflow History")
    print("=" * 20)

    db = MLDatabase()

    # Get workflow experiments (experiments with "workflow" in name)
    # This is a simplified query - in practice, you'd have a workflow_results table
    try:
        # For now, show recent experiments
        experiments_dir = Path("./workflow_results")
        if experiments_dir.exists():
            result_files = list(experiments_dir.glob("workflow_*.json"))
            result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            if result_files:
                print(f"Found {len(result_files)} completed workflows:")
                print()

                for i, result_file in enumerate(result_files[:10]):
                    try:
                        with open(result_file, 'r') as f:
                            workflow_data = json.load(f)

                        workflow_id = workflow_data.get('workflow_id', 'unknown')
                        status = "✅ Success" if workflow_data.get('success') else "❌ Failed"
                        timestamp = workflow_data.get('timestamp', 'unknown')

                        print(f"{i+1}. {result_file.name}")
                        print(f"   {status}")
                        print(f"   🕐 {timestamp}")

                        if workflow_data.get('success'):
                            summary = workflow_data.get('summary', {})
                            models = summary.get('models_trained', 0)
                            data = summary.get('data_samples', 0)
                            print(f"   🤖 {models} models, 📊 {data} samples")

                        print()

                    except Exception as e:
                        print(f"❌ Error reading {result_file.name}: {e}")

            else:
                print("No workflow results found.")
                print("Run a workflow first: python workflow_cli.py run")
        else:
            print("No workflow results directory found.")

    except Exception as e:
        print(f"❌ Error accessing workflow history: {e}")


def show_workflow_status(args):
    """Show current workflow status."""
    print("📊 Workflow Status")
    print("=" * 18)

    # Check for running workflow results files
    results_dir = Path("./workflow_results")
    if results_dir.exists():
        result_files = list(results_dir.glob("workflow_*.json"))
        if result_files:
            # Get most recent
            latest = max(result_files, key=lambda x: x.stat().st_mtime)

            try:
                with open(latest, 'r') as f:
                    workflow_data = json.load(f)

                print(f"Latest Workflow: {workflow_data.get('workflow_id', 'unknown')}")
                print(f"Status: {'✅ Completed' if workflow_data.get('success') else '❌ Failed'}")
                print(f"Timestamp: {workflow_data.get('timestamp', 'unknown')}")

                if workflow_data.get('success'):
                    summary = workflow_data.get('summary', {})
                    print(f"Duration: {summary.get('total_duration', 0):.1f} seconds")
                    print(f"Models Trained: {summary.get('models_trained', 0)}")
                    print(f"Data Samples: {summary.get('data_samples', 0)}")

            except Exception as e:
                print(f"Error reading latest workflow: {e}")
        else:
            print("No workflows found. Run: python workflow_cli.py run")
    else:
        print("No workflow results directory found.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Automated ML Workflow CLI - Complete ML pipelines from data to deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete workflow
  python workflow_cli.py run --days-back 365 --models MarketPredictor EdgeDetector

  # Quick test run
  python workflow_cli.py run --days-back 90 --yes

  # Check workflow status
  python workflow_cli.py status

  # View workflow history
  python workflow_cli.py history

  # Custom configuration
  python workflow_cli.py run --days-back 180 --min-volume 10000 --output-dir ./my_results
        """
    )

    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run complete automated ML workflow')
    run_parser.add_argument('--days-back', type=int, default=180,
                           help='Days of historical data to collect')
    run_parser.add_argument('--min-volume', type=int, default=5000,
                           help='Minimum market volume threshold')
    run_parser.add_argument('--models', nargs='+',
                           default=['MarketPredictor', 'EdgeDetector'],
                           help='ML models to train')
    run_parser.add_argument('--output-dir', type=str, default='./workflow_results',
                           help='Output directory for results')
    run_parser.add_argument('--yes', '-y', action='store_true',
                           help='Skip confirmation prompt')
    run_parser.add_argument('--save-results', action='store_true',
                           help='Save detailed results to JSON file')

    # Status command
    status_parser = subparsers.add_parser('status', help='Show current workflow status')

    # History command
    history_parser = subparsers.add_parser('history', help='Show workflow execution history')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == 'run':
            run_workflow(args)
        elif args.command == 'status':
            show_workflow_status(args)
        elif args.command == 'history':
            show_workflow_history(args)

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

