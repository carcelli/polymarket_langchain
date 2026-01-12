#!/usr/bin/env python3
"""
Standalone ML Workflow

A self-contained ML workflow that doesn't depend on complex ML strategy imports.
Demonstrates the complete pipeline from data collection to results storage.
"""

import sys
import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parents[2] / "src"))

# Direct imports to avoid dependency issues
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents', 'automl'))


class StandaloneMLWorkflow:
    """
    Self-contained ML workflow demonstrating the complete pipeline.

    This workflow:
    1. Creates its own simple database operations
    2. Collects/generates market data
    3. Plans ML approach
    4. "Trains" simple models (simulated)
    5. Stores results and generates reports
    """

    def __init__(self, db_path: str = "data/standalone_ml.db"):
        self.db_path = db_path
        self.workflow_id = f"standalone_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._init_database()

    def _init_database(self):
        """Initialize a simple database for this workflow."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            # Simple workflow tracking
            conn.execute('''
                CREATE TABLE IF NOT EXISTS workflows (
                    workflow_id TEXT PRIMARY KEY,
                    status TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    config TEXT
                )
            ''')

            # Simple experiment tracking
            conn.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    workflow_id TEXT,
                    name TEXT,
                    phase TEXT,
                    status TEXT,
                    results TEXT,
                    created_at TEXT
                )
            ''')

            # Simple model tracking
            conn.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    name TEXT,
                    type TEXT,
                    metrics TEXT,
                    created_at TEXT
                )
            ''')

    def _save_to_db(self, table: str, data: Dict[str, Any]):
        """Save data to database."""
        with sqlite3.connect(self.db_path) as conn:
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?' for _ in data])
            values = tuple(data.values())

            conn.execute(f'''
                INSERT OR REPLACE INTO {table} ({columns})
                VALUES ({placeholders})
            ''', values)

    def run_workflow(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the complete standalone ML workflow.

        Args:
            config: Workflow configuration

        Returns:
            Workflow results
        """
        print("🚀 Standalone ML Workflow")
        print("=" * 30)
        print(f"Workflow ID: {self.workflow_id}")
        print(f"Database: {self.db_path}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        config = config or {'days_back': 90, 'min_volume': 1000}

        # Save workflow start
        self._save_to_db('workflows', {
            'workflow_id': self.workflow_id,
            'status': 'running',
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'config': json.dumps(config)
        })

        results = {}

        try:
            # Phase 1: Data Collection
            print("📊 Phase 1: Data Collection")
            print("-" * 30)
            data_results = self._collect_data(config)
            results['data_collection'] = data_results

            # Phase 2: ML Planning
            print("\\n🧠 Phase 2: ML Planning")
            print("-" * 20)
            planning_results = self._plan_ml_approach(data_results)
            results['ml_planning'] = planning_results

            # Phase 3: Model Training
            print("\\n🤖 Phase 3: Model Training")
            print("-" * 25)
            training_results = self._train_models(data_results, planning_results)
            results['model_training'] = training_results

            # Phase 4: Results & Reporting
            print("\\n📋 Phase 4: Results & Reporting")
            print("-" * 32)
            reporting_results = self._generate_report(results)
            results['reporting'] = reporting_results

            # Update workflow as completed
            self._save_to_db('workflows', {
                'workflow_id': self.workflow_id,
                'status': 'completed',
                'start_time': datetime.now().isoformat(),  # Will be updated
                'end_time': datetime.now().isoformat(),
                'config': json.dumps(config)
            })

            # Success summary
            print("\\n🎉 Standalone ML Workflow Completed Successfully!")
            print("=" * 55)
            print(f"✅ Data collected: {data_results.get('samples', 0)} samples")
            print(f"✅ Strategies planned: {len(planning_results.get('strategies', []))}")
            print(f"✅ Models trained: {training_results.get('models_trained', 0)}")
            print(f"✅ Results stored in: {self.db_path}")

            return {
                'workflow_id': self.workflow_id,
                'status': 'completed',
                'success': True,
                'results': results,
                'database_path': self.db_path,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"\\n❌ Workflow failed: {e}")

            # Update workflow as failed
            self._save_to_db('workflows', {
                'workflow_id': self.workflow_id,
                'status': 'failed',
                'start_time': datetime.now().isoformat(),
                'end_time': datetime.now().isoformat(),
                'config': json.dumps(config)
            })

            return {
                'workflow_id': self.workflow_id,
                'status': 'failed',
                'success': False,
                'error': str(e),
                'results': results,
                'database_path': self.db_path,
                'timestamp': datetime.now().isoformat()
            }

    def _collect_data(self, config: Dict) -> Dict[str, Any]:
        """Collect market data."""
        days_back = config.get('days_back', 90)
        min_volume = config.get('min_volume', 1000)

        print(f"Collecting {days_back} days of market data (min volume: ${min_volume:,})...")

        # Create experiment for data collection
        experiment_id = f"exp_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Generate mock market data (simulates Polymarket API)
        np.random.seed(42)
        mock_data = []

        for i in range(200):  # Generate 200 mock markets
            volume = np.random.exponential(5000) + min_volume
            yes_price = np.random.beta(2, 2)  # Concentrated around 0.5
            category = np.random.choice(['politics', 'sports', 'crypto', 'economics'], p=[0.4, 0.3, 0.2, 0.1])

            # Simulate resolved markets with some correlation
            resolved = np.random.choice([True, False], p=[0.75, 0.25])
            actual_outcome = None
            if resolved:
                # Some correlation between price and outcome
                true_prob = yes_price + np.random.normal(0, 0.1)
                true_prob = np.clip(true_prob, 0.05, 0.95)
                actual_outcome = 1 if np.random.random() < true_prob else 0

            mock_data.append({
                'market_id': f'market_{i:04d}',
                'question': f'Sample market question {i} about {category}?',
                'category': category,
                'volume': volume,
                'yes_price': yes_price,
                'no_price': 1 - yes_price,
                'liquidity': volume * 0.05,  # 5% liquidity ratio
                'resolved': resolved,
                'actual_outcome': actual_outcome,
                'created_days_ago': np.random.randint(1, days_back)
            })

        # Convert to DataFrame and apply filters
        df = pd.DataFrame(mock_data)
        df = df[df['volume'] >= min_volume]

        # Add engineered features
        df['volume_log'] = np.log(df['volume'] + 1)
        df['price_distance'] = abs(df['yes_price'] - 0.5)
        df['high_volume'] = df['volume'] > 10000
        df['will_resolve_yes'] = df['actual_outcome']  # For ML training

        # Convert numpy types to Python types for JSON serialization
        df = df.astype({
            'volume': 'float64',
            'yes_price': 'float64',
            'no_price': 'float64',
            'liquidity': 'float64',
            'volume_log': 'float64',
            'price_distance': 'float64',
            'high_volume': 'bool',
            'will_resolve_yes': 'float64'
        })

        # Calculate data quality metrics
        quality_score = 85  # Mock quality score
        missing_data_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100

        # Convert to Python types for JSON
        experiment_results = {
            'samples_collected': int(len(df)),
            'features_engineered': int(len(df.columns)),
            'quality_score': int(quality_score),
            'missing_data_percent': float(missing_data_pct),
            'categories': {k: int(v) for k, v in df['category'].value_counts().to_dict().items()},
            'avg_volume': float(df['volume'].mean()),
            'resolved_markets': int(df['resolved'].sum())
        }

        self._save_to_db('experiments', {
            'experiment_id': experiment_id,
            'workflow_id': self.workflow_id,
            'name': 'Data Collection',
            'phase': 'data_collection',
            'status': 'completed',
            'results': json.dumps(experiment_results),
            'created_at': datetime.now().isoformat()
        })

        print(f"✅ Collected {len(df)} market samples")
        print(f"   📊 Quality score: {quality_score}/100")
        print(f"   📈 Average volume: ${df['volume'].mean():,.0f}")
        print(f"   🎯 Resolved markets: {df['resolved'].sum()}")

        return {
            'experiment_id': experiment_id,
            'samples': len(df),
            'features': len(df.columns),
            'quality_score': quality_score,
            'categories': df['category'].value_counts().to_dict(),
            'dataframe_sample': df.head(3).to_dict('records')
        }

    def _plan_ml_approach(self, data_results: Dict) -> Dict[str, Any]:
        """Plan ML approach based on data."""
        samples = data_results.get('samples', 0)
        quality_score = data_results.get('quality_score', 0)
        categories = data_results.get('categories', {})

        print("Planning ML strategy based on data characteristics...")

        # Create experiment for planning
        experiment_id = f"exp_planning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Plan strategies based on data characteristics
        strategies = []

        if samples > 100 and quality_score > 70:
            # High-quality data - can use advanced strategies
            strategies.extend([
                {
                    'name': 'Random Forest Predictor',
                    'type': 'ensemble_classification',
                    'description': 'Use random forest for outcome prediction',
                    'features': ['volume_log', 'price_distance', 'category', 'liquidity'],
                    'estimated_accuracy': '70-80%',
                    'complexity': 'medium'
                },
                {
                    'name': 'Volume-Based Regressor',
                    'type': 'regression',
                    'description': 'Predict price movements using volume features',
                    'features': ['volume', 'liquidity', 'high_volume'],
                    'estimated_accuracy': 'moderate correlation',
                    'complexity': 'low'
                },
                {
                    'name': 'Category-Specific Model',
                    'type': 'multi_model',
                    'description': 'Separate models for politics, sports, crypto',
                    'features': ['volume_log', 'price_distance', 'liquidity'],
                    'estimated_accuracy': '75-85%',
                    'complexity': 'high'
                }
            ])
        elif samples > 50:
            # Medium-quality data - simpler strategies
            strategies.extend([
                {
                    'name': 'Simple Price Predictor',
                    'type': 'baseline_classification',
                    'description': 'Basic prediction using current prices',
                    'features': ['yes_price', 'price_distance'],
                    'estimated_accuracy': '60-70%',
                    'complexity': 'low'
                },
                {
                    'name': 'Volume Threshold Strategy',
                    'type': 'rule_based',
                    'description': 'Simple rules based on volume thresholds',
                    'features': ['volume', 'high_volume'],
                    'estimated_accuracy': '55-65%',
                    'complexity': 'very_low'
                }
            ])
        else:
            # Low-quality data - very basic approach
            strategies.append({
                'name': 'Market Average Baseline',
                'type': 'baseline',
                'description': 'Use market average as prediction',
                'features': [],
                'estimated_accuracy': '50%',
                'complexity': 'minimal'
            })

        # Evaluation plan
        evaluation_plan = {
            'validation_method': 'train_test_split' if samples > 100 else 'simple_holdout',
            'test_size': 0.2,
            'metrics': ['accuracy', 'precision', 'recall', 'f1'],
            'cross_validation': samples > 150
        }

        # Risk assessment
        risk_factors = []
        if samples < 100:
            risk_factors.append("Limited data may lead to overfitting")
        if quality_score < 80:
            risk_factors.append("Data quality issues may affect model performance")
        if len(categories) < 3:
            risk_factors.append("Limited category diversity")

        planning_results = {
            'strategies_count': len(strategies),
            'strategies': strategies,
            'evaluation_plan': evaluation_plan,
            'risk_factors': risk_factors,
            'data_characteristics': {
                'samples': samples,
                'quality_score': quality_score,
                'categories': categories
            }
        }

        self._save_to_db('experiments', {
            'experiment_id': experiment_id,
            'workflow_id': self.workflow_id,
            'name': 'ML Planning',
            'phase': 'ml_planning',
            'status': 'completed',
            'results': json.dumps(planning_results),
            'created_at': datetime.now().isoformat()
        })

        print(f"✅ Planned {len(strategies)} ML strategies")
        for strategy in strategies:
            print(f"   • {strategy['name']}: {strategy['estimated_accuracy']} accuracy")

        if risk_factors:
            print(f"⚠️ Risk factors identified: {len(risk_factors)}")
            for risk in risk_factors:
                print(f"   • {risk}")

        return {
            'experiment_id': experiment_id,
            'strategies': strategies,
            'evaluation_plan': evaluation_plan,
            'risk_factors': risk_factors
        }

    def _train_models(self, data_results: Dict, planning_results: Dict) -> Dict[str, Any]:
        """Train models based on planning."""
        strategies = planning_results.get('strategies', [])

        print(f"Training {len(strategies)} ML models...")

        # Create experiment for training
        experiment_id = f"exp_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        training_results = []
        models_trained = 0

        for strategy in strategies:
            print(f"   🏗️ Training {strategy['name']}...")

            # Simulate model training
            import time
            time.sleep(0.2)  # Simulate training time

            # Mock training results based on strategy complexity
            complexity_bonus = {'very_low': 0, 'low': 0.05, 'medium': 0.1, 'high': 0.15}
            base_accuracy = 0.55 + complexity_bonus.get(strategy.get('complexity', 'low'), 0)

            # Add some randomness
            accuracy = base_accuracy + np.random.normal(0, 0.05)
            accuracy = np.clip(accuracy, 0.5, 0.9)

            mock_metrics = {
                'accuracy': accuracy,
                'precision': accuracy - 0.05 + np.random.normal(0, 0.02),
                'recall': accuracy + 0.02 + np.random.normal(0, 0.02),
                'f1': accuracy + np.random.normal(0, 0.02),
                'training_samples': data_results.get('samples', 0),
                'features_used': len(strategy.get('features', [])),
                'training_time_seconds': 0.2
            }

            # Create model record
            model_id = f"model_{strategy['name'].replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            model_info = {
                'model_id': model_id,
                'experiment_id': experiment_id,
                'name': strategy['name'],
                'type': strategy['type'],
                'metrics': json.dumps(mock_metrics),
                'created_at': datetime.now().isoformat()
            }

            self._save_to_db('models', model_info)

            training_results.append({
                'model_id': model_id,
                'strategy_name': strategy['name'],
                'metrics': mock_metrics,
                'success': True
            })

            models_trained += 1
            print(".1%")

        training_summary = {
            'models_trained': models_trained,
            'total_strategies': len(strategies),
            'avg_accuracy': np.mean([r['metrics']['accuracy'] for r in training_results]),
            'best_accuracy': max([r['metrics']['accuracy'] for r in training_results]),
            'training_results': training_results
        }

        self._save_to_db('experiments', {
            'experiment_id': experiment_id,
            'workflow_id': self.workflow_id,
            'name': 'Model Training',
            'phase': 'model_training',
            'status': 'completed',
            'results': json.dumps(training_summary),
            'created_at': datetime.now().isoformat()
        })

        print(f"\\n✅ Training completed: {models_trained}/{len(strategies)} models successful")

        return {
            'experiment_id': experiment_id,
            'models_trained': models_trained,
            'training_results': training_results,
            'summary': training_summary
        }

    def _generate_report(self, workflow_results: Dict) -> Dict[str, Any]:
        """Generate workflow report."""
        print("Generating comprehensive workflow report...")

        # Create experiment for reporting
        experiment_id = f"exp_reporting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Compile comprehensive report
        report = {
            'workflow_id': self.workflow_id,
            'generated_at': datetime.now().isoformat(),
            'execution_summary': {
                'total_phases': len(workflow_results),
                'phases_completed': len([r for r in workflow_results.values() if r]),
                'database_path': self.db_path
            },
            'data_insights': {
                'samples_collected': workflow_results.get('data_collection', {}).get('samples', 0),
                'quality_score': workflow_results.get('data_collection', {}).get('quality_score', 0),
                'categories': workflow_results.get('data_collection', {}).get('categories', {})
            },
            'ml_insights': {
                'strategies_planned': len(workflow_results.get('ml_planning', {}).get('strategies', [])),
                'models_trained': workflow_results.get('model_training', {}).get('models_trained', 0),
                'best_accuracy': workflow_results.get('model_training', {}).get('summary', {}).get('best_accuracy', 0)
            },
            'performance_summary': {
                'overall_success': True,
                'data_pipeline_efficiency': 'high',
                'ml_pipeline_efficiency': 'medium',
                'automation_level': 'full'
            },
            'recommendations': [
                "Consider collecting more diverse market data",
                "Implement automated model retraining pipelines",
                "Add more sophisticated feature engineering",
                "Set up model performance monitoring",
                "Integrate with additional data sources"
            ],
            'next_steps': [
                "Deploy best-performing models to production",
                "Set up automated daily data collection",
                "Implement model performance dashboards",
                "Add A/B testing for different strategies",
                "Consider ensemble methods for improved accuracy"
            ]
        }

        self._save_to_db('experiments', {
            'experiment_id': experiment_id,
            'workflow_id': self.workflow_id,
            'name': 'Results Reporting',
            'phase': 'reporting',
            'status': 'completed',
            'results': json.dumps(report),
            'created_at': datetime.now().isoformat()
        })

        # Save report to file
        report_file = f"workflow_report_{self.workflow_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print("✅ Comprehensive report generated")
        print(f"   📄 Report saved: {report_file}")
        print(f"   📊 Models trained: {report['ml_insights']['models_trained']}")
        print(".1%")
        print(f"   🎯 Categories: {len(report['data_insights']['categories'])}")

        return {
            'experiment_id': experiment_id,
            'report': report,
            'report_file': report_file
        }


def run_standalone_ml_workflow(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run the standalone ML workflow.

    Args:
        config: Workflow configuration

    Returns:
        Workflow results
    """
    workflow = StandaloneMLWorkflow()
    return workflow.run_workflow(config)


def show_workflow_database(db_path: str = "data/standalone_ml.db"):
    """Show contents of workflow database."""
    print(f"📊 Workflow Database Contents: {db_path}")
    print("=" * 50)

    if not os.path.exists(db_path):
        print("❌ Database file not found")
        return

    with sqlite3.connect(db_path) as conn:
        # Show workflows
        workflows = conn.execute("SELECT * FROM workflows ORDER BY start_time DESC").fetchall()
        print(f"\\n📋 Workflows ({len(workflows)}):")
        for workflow in workflows:
            print(f"   • {workflow[0]}: {workflow[1]} ({workflow[3] or 'Running'})")

        # Show experiments
        experiments = conn.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()
        print(f"\\n🧪 Experiments ({len(experiments)}):")
        for exp in experiments:
            print(f"   • {exp[0]}: {exp[2]} ({exp[4]})")

        # Show models
        models = conn.execute("SELECT * FROM models ORDER BY created_at DESC").fetchall()
        print(f"\\n🤖 Models ({len(models)}):")
        for model in models:
            metrics = json.loads(model[4]) if model[4] else {}
            accuracy = metrics.get('accuracy', 0)
            print(".1%")
if __name__ == "__main__":
    # Example usage
    config = {
        'days_back': 90,
        'min_volume': 1000
    }

    print("🤖 Starting Standalone ML Workflow")
    print("This workflow demonstrates:")
    print("1. 📊 Data collection and processing")
    print("2. 🧠 ML strategy planning")
    print("3. 🤖 Model training simulation")
    print("4. 📋 Results reporting and storage")
    print()

    results = run_standalone_ml_workflow(config)

    if results.get('success'):
        print("\\n🎉 Standalone ML Workflow completed successfully!")
        print(f"Workflow ID: {results['workflow_id']}")
        print(f"Database: {results['database_path']}")

        # Show database contents
        show_workflow_database(results['database_path'])

    else:
        print(f"\\n❌ Standalone ML Workflow failed: {results.get('error', 'Unknown error')}")

    print("\\n💡 To explore the database further, run:")
    print("python standalone_ml_workflow.py --show-db")
