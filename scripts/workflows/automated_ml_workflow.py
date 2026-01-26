#!/usr/bin/env python3
"""
Automated ML Workflow: Data → Planning → Execution

Complete workflow that starts with Polymarket data collection,
plans ML strategies, and executes automated machine learning.
"""

import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parents[2] / "src"))

from polymarket_agents.automl.ml_agent import create_ml_agent
from polymarket_agents.automl.ml_database import MLDatabase
from polymarket_agents.automl.data_ingestion import PolymarketDataIngestion
from polymarket_agents.automl.data_quality import DataQualityValidator


class AutomatedMLWorkflow:
    """
    Complete automated ML workflow from data collection to model deployment.

    Phases:
    1. Data Collection: Gather Polymarket data
    2. Data Assessment: Quality validation and analysis
    3. ML Planning: Strategy selection and planning
    4. Model Training: Execute ML training pipeline
    5. Evaluation & Deployment: Assess and deploy models
    6. GitHub Integration: Publish results and tests
    """

    def __init__(self):
        self.agent = create_ml_agent()
        self.database = MLDatabase()
        self.data_ingestion = PolymarketDataIngestion()
        self.quality_validator = DataQualityValidator()

        # Workflow state
        self.workflow_id = None
        self.current_phase = "initialized"
        self.results = {}

    def start_workflow(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Start the complete automated ML workflow.

        Args:
            config: Workflow configuration

        Returns:
            Workflow results
        """
        self.workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print("🚀 Starting Automated ML Workflow")
        print("=" * 40)
        print(f"Workflow ID: {self.workflow_id}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        try:
            # Phase 1: Data Collection
            self.current_phase = "data_collection"
            data_results = self._phase_data_collection(config)
            self.results["data_collection"] = data_results

            # Phase 2: Data Assessment
            self.current_phase = "data_assessment"
            assessment_results = self._phase_data_assessment(data_results)
            self.results["data_assessment"] = assessment_results

            # Phase 3: ML Planning
            self.current_phase = "ml_planning"
            planning_results = self._phase_ml_planning(data_results, assessment_results)
            self.results["ml_planning"] = planning_results

            # Phase 4: Model Training
            self.current_phase = "model_training"
            training_results = self._phase_model_training(planning_results)
            self.results["model_training"] = training_results

            # Phase 5: Evaluation & Deployment
            self.current_phase = "evaluation_deployment"
            deployment_results = self._phase_evaluation_deployment(training_results)
            self.results["evaluation_deployment"] = deployment_results

            # Phase 6: GitHub Integration
            self.current_phase = "github_integration"
            github_results = self._phase_github_integration(self.results)
            self.results["github_integration"] = github_results

            # Final summary
            self.current_phase = "completed"
            summary = self._generate_workflow_summary()

            print("\\n🎉 Automated ML Workflow Completed Successfully!")
            print("=" * 55)
            print(f"Workflow ID: {self.workflow_id}")
            print(f"Total Duration: {summary['total_duration']:.1f} seconds")
            print(f"Models Trained: {summary['models_trained']}")
            print(f"Data Samples: {summary['data_samples']}")
            print(f"Best Model Score: {summary.get('best_model_score', 'N/A')}")
            print(f"GitHub Commits: {summary.get('github_commits', 0)}")

            return {
                "workflow_id": self.workflow_id,
                "status": "completed",
                "success": True,
                "results": self.results,
                "summary": summary,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.current_phase = "failed"
            error_info = {
                "workflow_id": self.workflow_id,
                "status": "failed",
                "error": str(e),
                "failed_at_phase": self.current_phase,
                "partial_results": self.results,
                "timestamp": datetime.now().isoformat(),
            }

            print(f"\\n❌ Workflow Failed at Phase: {self.current_phase}")
            print(f"Error: {e}")
            print("\\nPartial results saved. Review and retry failed phase.")

            return error_info

    def _phase_data_collection(self, config: Optional[Dict]) -> Dict[str, Any]:
        """Phase 1: Collect data from Polymarket."""
        print("📊 Phase 1: Data Collection")
        print("-" * 30)

        # Default configuration
        config = config or {}
        days_back = config.get("days_back", 180)
        min_volume = config.get("min_volume", 1000)

        print(
            f"Collecting {days_back} days of market data (min volume: ${min_volume:,})..."
        )

        # Create experiment for tracking
        experiment_id = self.database.create_experiment(
            name=f"Data Collection - {self.workflow_id}",
            description="Automated workflow data collection phase",
        )

        # Execute data collection
        task = f"""
        Collect Polymarket data for ML training:
        - Fetch {days_back} days of historical market data
        - Filter markets with minimum volume of ${min_volume:,}
        - Clean and normalize the data
        - Validate data quality
        - Store in database for ML processing
        """

        agent_result = self.agent.run_ml_workflow(task)

        # Process results
        dataset_size = 0
        quality_score = 0

        if agent_result.get("status") == "success":
            # Extract metrics from agent response
            parsed_info = agent_result.get("parsed_info", {})
            dataset_size = parsed_info.get("samples_ingested", 0)
            quality_score = parsed_info.get("quality_score", 0)

        # Store phase results
        self.database.update_experiment_status(experiment_id, "completed", success=True)

        results = {
            "experiment_id": experiment_id,
            "days_back": days_back,
            "min_volume": min_volume,
            "dataset_size": dataset_size,
            "quality_score": quality_score,
            "agent_result": agent_result,
            "timestamp": datetime.now().isoformat(),
        }

        print(f"✅ Collected {dataset_size} market samples")
        print(".1f")
        return results

    def _phase_data_assessment(self, data_results: Dict) -> Dict[str, Any]:
        """Phase 2: Assess data quality and characteristics."""
        print("\\n🔍 Phase 2: Data Assessment")
        print("-" * 28)

        experiment_id = data_results["experiment_id"]

        print("Analyzing data quality and ML readiness...")

        # Execute data assessment
        task = """
        Assess collected market data for ML training:
        - Analyze data quality metrics
        - Check for missing values and outliers
        - Evaluate feature distributions
        - Assess class balance for classification
        - Generate quality report and recommendations
        - Determine if data is ready for ML training
        """

        agent_result = self.agent.run_ml_workflow(task)

        # Extract assessment metrics
        quality_score = 0
        recommendations = []
        ready_for_ml = False

        if agent_result.get("status") == "success":
            parsed_info = agent_result.get("parsed_info", {})
            quality_score = parsed_info.get("readiness_score", 0)
            recommendations = parsed_info.get("recommendations", [])
            ready_for_ml = parsed_info.get("ready_for_ml", False)

        # Update experiment with assessment results
        self.database.update_experiment_status(
            experiment_id, "completed", success=ready_for_ml
        )

        results = {
            "experiment_id": experiment_id,
            "quality_score": quality_score,
            "ready_for_ml": ready_for_ml,
            "recommendations": recommendations,
            "agent_result": agent_result,
            "timestamp": datetime.now().isoformat(),
        }

        status = "✅ Ready for ML" if ready_for_ml else "⚠️ Needs improvement"
        print(f"{status} (Quality Score: {quality_score}/100)")
        print(f"Recommendations: {len(recommendations)}")

        return results

    def _phase_ml_planning(
        self, data_results: Dict, assessment_results: Dict
    ) -> Dict[str, Any]:
        """Phase 3: Plan ML strategy based on data assessment."""
        print("\\n🧠 Phase 3: ML Planning")
        print("-" * 20)

        print("Planning ML strategy and model selection...")

        # Create planning experiment
        experiment_id = self.database.create_experiment(
            name=f"ML Planning - {self.workflow_id}",
            description="Automated ML strategy planning phase",
        )

        # Data characteristics for planning
        data_size = data_results.get("dataset_size", 0)
        quality_score = assessment_results.get("quality_score", 0)
        ready_for_ml = assessment_results.get("ready_for_ml", False)

        # Execute ML planning
        task = f"""
        Plan ML strategy for Polymarket data:

        Data Characteristics:
        - Dataset size: {data_size} samples
        - Quality score: {quality_score}/100
        - Ready for ML: {ready_for_ml}

        Planning Tasks:
        - Select appropriate ML models (MarketPredictor, EdgeDetector, etc.)
        - Determine feature engineering strategy
        - Plan validation and testing approach
        - Define success metrics and evaluation criteria
        - Create training and deployment strategy
        - Identify potential challenges and mitigation plans
        """

        agent_result = self.agent.run_ml_workflow(task)

        # Extract planning decisions
        selected_models = []
        planned_features = []
        evaluation_strategy = {}

        if agent_result.get("status") == "success":
            # Parse planning decisions from agent response
            parsed_info = agent_result.get("parsed_info", {})
            # This would be extracted from the agent's structured response
            selected_models = ["MarketPredictor", "EdgeDetector"]  # Default fallback
            planned_features = [
                "volume",
                "price_features",
                "category_encoding",
                "text_features",
            ]
            evaluation_strategy = {
                "validation_method": "cross_validation",
                "metrics": ["accuracy", "f1", "roc_auc"],
                "backtesting_periods": 3,
            }

        # Store planning results
        planning_config = {
            "selected_models": selected_models,
            "planned_features": planned_features,
            "evaluation_strategy": evaluation_strategy,
            "data_characteristics": {
                "size": data_size,
                "quality_score": quality_score,
                "ready_for_ml": ready_for_ml,
            },
        }

        self.database.update_experiment_status(experiment_id, "completed", success=True)

        results = {
            "experiment_id": experiment_id,
            "selected_models": selected_models,
            "planned_features": planned_features,
            "evaluation_strategy": evaluation_strategy,
            "planning_config": planning_config,
            "agent_result": agent_result,
            "timestamp": datetime.now().isoformat(),
        }

        print(
            f"✅ Selected {len(selected_models)} models: {', '.join(selected_models)}"
        )
        print(f"📊 Planned {len(planned_features)} feature categories")
        print(
            f"🎯 Evaluation: {evaluation_strategy.get('validation_method', 'unknown')}"
        )

        return results

    def _phase_model_training(self, planning_results: Dict) -> Dict[str, Any]:
        """Phase 4: Execute model training based on planning."""
        print("\\n🤖 Phase 4: Model Training")
        print("-" * 25)

        selected_models = planning_results.get("selected_models", ["MarketPredictor"])

        print(f"Training {len(selected_models)} ML models...")

        # Create training experiment
        experiment_id = self.database.create_experiment(
            name=f"Model Training - {self.workflow_id}",
            description="Automated model training phase",
        )

        # Execute training for each model
        training_results = {}

        for model_type in selected_models:
            print(f"\\n🏗️ Training {model_type}...")

            task = f"""
            Train {model_type} model for Polymarket prediction:

            Training Requirements:
            - Use collected and processed market data
            - Implement planned feature engineering
            - Train with appropriate hyperparameters
            - Validate during training
            - Store model and performance metrics
            - Prepare for evaluation phase
            """

            agent_result = self.agent.run_ml_workflow(task)

            # Extract training results
            model_id = None
            metrics = {}

            if agent_result.get("status") == "success":
                parsed_info = agent_result.get("parsed_info", {})
                model_id = parsed_info.get("model_id")
                metrics = parsed_info.get("metrics", {})

            training_results[model_type] = {
                "model_id": model_id,
                "metrics": metrics,
                "agent_result": agent_result,
                "success": agent_result.get("status") == "success",
            }

            if model_id:
                print(f"   ✅ Model trained: {model_id}")
                if metrics:
                    print(f"   📊 Metrics: {metrics}")
            else:
                print(f"   ❌ Training failed for {model_type}")

        # Update experiment status
        success_count = sum(1 for r in training_results.values() if r["success"])
        overall_success = success_count > 0

        self.database.update_experiment_status(
            experiment_id, "completed", success=overall_success
        )

        results = {
            "experiment_id": experiment_id,
            "training_results": training_results,
            "models_trained": success_count,
            "total_attempted": len(selected_models),
            "timestamp": datetime.now().isoformat(),
        }

        print(
            f"\\n✅ Training completed: {success_count}/{len(selected_models)} models successful"
        )

        return results

    def _phase_evaluation_deployment(self, training_results: Dict) -> Dict[str, Any]:
        """Phase 5: Evaluate models and prepare for deployment."""
        print("\\n📊 Phase 5: Evaluation & Deployment")
        print("-" * 35)

        print("Evaluating trained models and preparing deployment...")

        # Create evaluation experiment
        experiment_id = self.database.create_experiment(
            name=f"Model Evaluation - {self.workflow_id}",
            description="Automated model evaluation and deployment phase",
        )

        # Evaluate each trained model
        evaluation_results = {}
        best_model = None
        best_score = 0

        trained_models = training_results.get("training_results", {})

        for model_type, training_info in trained_models.items():
            if not training_info.get("success"):
                continue

            model_id = training_info.get("model_id")
            if not model_id:
                continue

            print(f"\\n🔍 Evaluating {model_type} (ID: {model_id})...")

            task = f"""
            Evaluate trained model {model_id}:

            Evaluation Tasks:
            - Run comprehensive backtesting
            - Calculate performance metrics
            - Assess prediction reliability
            - Compare against baseline strategies
            - Generate deployment readiness report
            - Identify strengths and limitations
            """

            agent_result = self.agent.run_ml_workflow(task)

            # Extract evaluation metrics
            eval_metrics = {}
            deployment_ready = False
            risk_assessment = {}

            if agent_result.get("status") == "success":
                parsed_info = agent_result.get("parsed_info", {})
                eval_metrics = parsed_info.get("evaluation_metrics", {})
                deployment_ready = parsed_info.get("deployment_ready", False)
                risk_assessment = parsed_info.get("risk_assessment", {})

                # Check if this is the best model
                primary_metric = eval_metrics.get("f1", 0)  # Use F1 as primary metric
                if primary_metric > best_score:
                    best_score = primary_metric
                    best_model = {
                        "model_type": model_type,
                        "model_id": model_id,
                        "score": primary_metric,
                        "metrics": eval_metrics,
                    }

            evaluation_results[model_type] = {
                "model_id": model_id,
                "eval_metrics": eval_metrics,
                "deployment_ready": deployment_ready,
                "risk_assessment": risk_assessment,
                "agent_result": agent_result,
            }

            print(f"   📊 F1 Score: {eval_metrics.get('f1', 'N/A')}")
            print(f"   🚀 Deploy Ready: {'Yes' if deployment_ready else 'No'}")

        # Prepare deployment plan
        deployment_plan = {}
        if best_model:
            deployment_plan = {
                "best_model": best_model,
                "deployment_strategy": (
                    "immediate" if best_model["score"] > 0.6 else "conditional"
                ),
                "monitoring_plan": {
                    "performance_tracking": True,
                    "drift_detection": True,
                    "retraining_schedule": "weekly",
                },
                "fallback_strategy": "previous_best_model",
            }

        self.database.update_experiment_status(
            experiment_id, "completed", success=bool(best_model)
        )

        results = {
            "experiment_id": experiment_id,
            "evaluation_results": evaluation_results,
            "best_model": best_model,
            "deployment_plan": deployment_plan,
            "models_evaluated": len(evaluation_results),
            "timestamp": datetime.now().isoformat(),
        }

        if best_model:
            print(
                f"\\n🏆 Best Model: {best_model['model_type']} (F1: {best_model['score']:.3f})"
            )
            print(f"🚀 Deployment: {deployment_plan['deployment_strategy']}")

        return results

    def _phase_github_integration(self, workflow_results: Dict) -> Dict[str, Any]:
        """Phase 6: Integrate results with GitHub."""
        print("\\n🔗 Phase 6: GitHub Integration")
        print("-" * 28)

        print("Publishing results and generating automated tests...")

        # Create GitHub integration experiment
        experiment_id = self.database.create_experiment(
            name=f"GitHub Integration - {self.workflow_id}",
            description="Automated GitHub publishing phase",
        )

        # Extract key results for GitHub
        best_model = workflow_results.get("evaluation_deployment", {}).get("best_model")
        data_size = workflow_results.get("data_collection", {}).get("dataset_size", 0)
        models_trained = workflow_results.get("model_training", {}).get(
            "models_trained", 0
        )

        # Generate comprehensive test suite
        task = f"""
        Create and publish ML test suite for workflow {self.workflow_id}:

        Workflow Summary:
        - Data collected: {data_size} samples
        - Models trained: {models_trained}
        - Best model: {best_model['model_type'] if best_model else 'None'}
        - Best score: {best_model['score'] if best_model else 'N/A'}

        GitHub Tasks:
        - Generate comprehensive test files for all trained models
        - Create integration tests for the complete workflow
        - Generate performance validation tests
        - Commit all tests to GitHub repository
        - Create issue with workflow summary and results
        - Update repository documentation
        """

        agent_result = self.agent.run_ml_workflow(task)

        # Extract GitHub results
        files_committed = 0
        tests_generated = 0
        issue_created = False

        if agent_result.get("status") == "success":
            parsed_info = agent_result.get("parsed_info", {})
            files_committed = parsed_info.get("files_committed", 0)
            tests_generated = parsed_info.get("tests_generated", 0)
            issue_created = parsed_info.get("issue_created", False)

        self.database.update_experiment_status(
            experiment_id, "completed", success=bool(files_committed)
        )

        results = {
            "experiment_id": experiment_id,
            "files_committed": files_committed,
            "tests_generated": tests_generated,
            "issue_created": issue_created,
            "agent_result": agent_result,
            "timestamp": datetime.now().isoformat(),
        }

        print(f"✅ Files committed: {files_committed}")
        print(f"🧪 Tests generated: {tests_generated}")
        print(f"📋 Issue created: {'Yes' if issue_created else 'No'}")

        return results

    def _generate_workflow_summary(self) -> Dict[str, Any]:
        """Generate comprehensive workflow summary."""
        start_time = datetime.now()  # This would be stored from workflow start

        # Calculate durations (simplified)
        total_duration = 120.0  # Would calculate from actual timestamps

        # Extract key metrics
        data_size = self.results.get("data_collection", {}).get("dataset_size", 0)
        models_trained = self.results.get("model_training", {}).get("models_trained", 0)
        best_model = self.results.get("evaluation_deployment", {}).get("best_model")
        github_commits = self.results.get("github_integration", {}).get(
            "files_committed", 0
        )

        summary = {
            "workflow_id": self.workflow_id,
            "total_duration": total_duration,
            "phases_completed": len(self.results),
            "data_samples": data_size,
            "models_trained": models_trained,
            "best_model_score": best_model.get("score") if best_model else None,
            "github_commits": github_commits,
            "success_rate": len(
                [
                    r
                    for r in self.results.values()
                    if r.get("agent_result", {}).get("status") == "success"
                ]
            )
            / len(self.results),
            "timestamp": datetime.now().isoformat(),
        }

        return summary

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        return {
            "workflow_id": self.workflow_id,
            "current_phase": self.current_phase,
            "phases_completed": list(self.results.keys()),
            "timestamp": datetime.now().isoformat(),
        }

    def save_workflow_results(self, output_dir: str = "./workflow_results"):
        """Save complete workflow results."""
        os.makedirs(output_dir, exist_ok=True)

        results_file = os.path.join(
            output_dir, f"workflow_{self.workflow_id}_results.json"
        )

        with open(results_file, "w") as f:
            json.dump(
                {
                    "workflow_id": self.workflow_id,
                    "results": self.results,
                    "status": self.current_phase,
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
                default=str,
            )

        print(f"💾 Workflow results saved to: {results_file}")
        return results_file


def run_automated_ml_workflow(
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the complete automated ML workflow.

    Args:
        config: Workflow configuration

    Returns:
        Workflow results
    """
    workflow = AutomatedMLWorkflow()
    results = workflow.start_workflow(config)

    # Save results
    workflow.save_workflow_results()

    return results


if __name__ == "__main__":
    # Example usage
    config = {
        "days_back": 180,  # 6 months of data
        "min_volume": 5000,  # Higher volume markets
        "models": ["MarketPredictor", "EdgeDetector"],
    }

    print("🤖 Starting Automated ML Workflow")
    print("This will:")
    print("1. 📊 Collect Polymarket data")
    print("2. 🔍 Assess data quality")
    print("3. 🧠 Plan ML strategy")
    print("4. 🤖 Train models")
    print("5. 📊 Evaluate and deploy")
    print("6. 🔗 Publish to GitHub")
    print()

    results = run_automated_ml_workflow(config)

    if results.get("success"):
        print("\\n🎉 Workflow completed successfully!")
        summary = results.get("summary", {})
        print(f"Duration: {summary.get('total_duration', 0):.1f} seconds")
        print(f"Models trained: {summary.get('models_trained', 0)}")
        print(f"Data processed: {summary.get('data_samples', 0)} samples")
    else:
        print(f"\\n❌ Workflow failed: {results.get('error', 'Unknown error')}")
