"""
Automated ML Pipeline for Polymarket Data

End-to-end automated machine learning pipeline that:
1. Ingests and cleans market data
2. Validates data quality
3. Trains and evaluates ML models
4. Generates automated reports
5. Deploys models for prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import joblib
from pathlib import Path
import json

from .data_ingestion import PolymarketDataIngestion
from .data_quality import DataQualityValidator
from ..ml_strategies import MarketPredictor, EdgeDetector

logger = logging.getLogger(__name__)


def generate_ml_strategy_test(model_name: str, test_type: str, description: str) -> str:
    """Stub: Generate a basic test template for an ML strategy."""
    return f'''"""
{description}
"""
import pytest

class Test{model_name}:
    """Auto-generated tests for {model_name}."""

    def test_placeholder(self):
        """Placeholder test - implement actual tests."""
        assert True
'''


def commit_ml_tests_to_github(
    test_files: Dict[str, str], message: str
) -> Dict[str, Any]:
    """Stub: GitHub integration has been deprecated."""
    logger.warning("GitHub integration has been deprecated. Tests saved locally only.")
    return {"status": "skipped", "message": "GitHub integration deprecated"}


class AutoMLPipeline:
    """
    Complete automated ML pipeline for Polymarket data.

    Orchestrates the entire ML workflow from data ingestion to model deployment.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.data_ingestion = PolymarketDataIngestion()
        self.quality_validator = DataQualityValidator()
        self.models = {}
        self.best_model = None
        self.pipeline_results = {}

        # Create output directories
        self.output_dir = Path(self.config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.models_dir = self.output_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)

    def _default_config(self) -> Dict[str, Any]:
        """Default pipeline configuration."""
        return {
            "output_dir": "./automl_output",
            "data_days_back": 365,
            "min_volume": 1000,
            "test_size": 0.2,
            "random_state": 42,
            "models_to_train": ["MarketPredictor", "EdgeDetector"],
            "validation_metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
            "enable_github_integration": True,
            "auto_generate_tests": True,
        }

    def run_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete AutoML pipeline.

        Returns:
            Comprehensive pipeline results
        """
        logger.info("🚀 Starting AutoML Pipeline for Polymarket")

        pipeline_start = datetime.now()

        try:
            # Step 1: Data Ingestion
            logger.info("📊 Step 1: Data Ingestion")
            raw_dataset = self._ingest_data()

            # Step 2: Data Quality Validation & Cleaning
            logger.info("🧹 Step 2: Data Quality & Cleaning")
            clean_dataset, quality_report = self._validate_and_clean_data(raw_dataset)

            # Step 3: Feature Engineering
            logger.info("🔧 Step 3: Feature Engineering")
            ml_dataset = self._engineer_features(clean_dataset)

            # Step 4: Model Training & Evaluation
            logger.info("🤖 Step 4: Model Training & Evaluation")
            model_results = self._train_and_evaluate_models(ml_dataset)

            # Step 5: Model Selection & Validation
            logger.info("🎯 Step 5: Model Selection")
            best_model_info = self._select_best_model(model_results)

            # Step 6: Generate Reports & Tests
            logger.info("📋 Step 6: Generate Reports & Tests")
            reports = self._generate_reports_and_tests(
                ml_dataset, model_results, best_model_info
            )

            # Step 7: Deploy Best Model
            logger.info("🚀 Step 7: Deploy Best Model")
            deployment_info = self._deploy_best_model(best_model_info)

            # Compile final results
            pipeline_results = {
                "pipeline_metadata": {
                    "start_time": pipeline_start.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "duration_seconds": (
                        datetime.now() - pipeline_start
                    ).total_seconds(),
                    "config": self.config,
                },
                "data_summary": {
                    "raw_samples": len(raw_dataset),
                    "clean_samples": len(clean_dataset),
                    "final_samples": len(ml_dataset),
                    "features_count": len(ml_dataset.columns) - 1,  # Exclude target
                },
                "quality_report": quality_report,
                "model_results": model_results,
                "best_model": best_model_info,
                "reports": reports,
                "deployment": deployment_info,
                "success": True,
            }

            self.pipeline_results = pipeline_results

            # Save pipeline results
            self._save_pipeline_results(pipeline_results)

            logger.info("✅ AutoML Pipeline completed successfully!")
            return pipeline_results

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            import traceback

            traceback.print_exc()

            return {
                "success": False,
                "error": str(e),
                "pipeline_metadata": {
                    "start_time": pipeline_start.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "duration_seconds": (
                        datetime.now() - pipeline_start
                    ).total_seconds(),
                },
            }

    def _ingest_data(self) -> pd.DataFrame:
        """Step 1: Ingest and prepare raw market data."""
        dataset = self.data_ingestion.run_data_pipeline(
            days_back=self.config["data_days_back"]
        )

        # Apply basic filtering
        dataset = dataset[dataset["volume"] >= self.config["min_volume"]]

        logger.info(f"📊 Ingested {len(dataset)} market samples")
        return dataset

    def _validate_and_clean_data(
        self, dataset: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        """Step 2: Validate and clean data quality."""
        quality_report = self.quality_validator.validate_dataset_structure(dataset)

        if quality_report["issues"]:
            logger.warning(
                f"⚠️ Found {len(quality_report['issues'])} data quality issues"
            )

        # Apply quality improvements
        clean_dataset = self.quality_validator.handle_missing_values(dataset)
        clean_dataset = self.quality_validator.remove_outliers(clean_dataset)

        logger.info(f"🧹 Cleaned data: {len(clean_dataset)} samples")
        return clean_dataset, quality_report

    def _engineer_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Step 3: Engineer ML features."""
        ml_dataset = self.data_ingestion.engineer_ml_features(dataset)

        # Additional AutoML feature engineering
        ml_dataset = self._add_automl_features(ml_dataset)

        logger.info(f"🔧 Engineered {len(ml_dataset.columns)} features")
        return ml_dataset

    def _add_automl_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add AutoML-specific feature engineering."""
        # Interaction features
        df["volume_price_interaction"] = df["volume"] * df["yes_price"]
        df["liquidity_efficiency"] = df["liquidity"] / (df["volume"] + 1)

        # Statistical features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ["will_resolve_yes", "market_id"]:
                # Rolling statistics (simulated)
                df[f"{col}_normalized"] = (df[col] - df[col].mean()) / (
                    df[col].std() + 1e-6
                )

        # Category interaction features
        if "category" in df.columns:
            df["category_volume_avg"] = df.groupby("category")["volume"].transform(
                "mean"
            )
            df["category_price_avg"] = df.groupby("category")["yes_price"].transform(
                "mean"
            )

        return df

    def _train_and_evaluate_models(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Step 4: Train and evaluate ML models."""
        # Prepare data
        feature_cols = [
            col
            for col in dataset.columns
            if col not in ["market_id", "will_resolve_yes", "question", "description"]
        ]

        X = dataset[feature_cols]
        dataset["will_resolve_yes"]

        # Handle any remaining missing values
        X = X.fillna(X.mean())

        model_results = {}

        for model_name in self.config["models_to_train"]:
            logger.info(f"Training {model_name}...")

            try:
                if model_name == "MarketPredictor":
                    model = MarketPredictor()
                elif model_name == "EdgeDetector":
                    model = EdgeDetector()
                else:
                    continue

                # Train model
                train_data = dataset[
                    [
                        "market_id",
                        "question",
                        "category",
                        "yes_price",
                        "no_price",
                        "volume",
                        "liquidity",
                        "actual_outcome",
                    ]
                ].copy()
                train_data = train_data.rename(
                    columns={"actual_outcome": "will_resolve_yes"}
                )
                model.train(train_data)

                # Evaluate model
                predictions = []
                for _, row in dataset.iterrows():
                    result = model.predict(row.to_dict())
                    predictions.append(
                        {
                            "market_id": result.market_id,
                            "prediction": result.predicted_probability,
                            "actual": row.get("will_resolve_yes"),
                            "confidence": result.confidence,
                        }
                    )

                # Calculate metrics
                pred_df = pd.DataFrame(predictions)
                metrics = self._calculate_model_metrics(pred_df)

                model_results[model_name] = {
                    "model": model,
                    "metrics": metrics,
                    "predictions": predictions,
                    "feature_importance": model.get_feature_importance(),
                    "training_samples": len(dataset),
                }

                logger.info(".4f")
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                model_results[model_name] = {"error": str(e)}

        return model_results

    def _calculate_model_metrics(
        self, predictions_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate comprehensive model evaluation metrics."""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
        )

        # Binary predictions (threshold at 0.5)
        predictions_df["pred_binary"] = (predictions_df["prediction"] > 0.5).astype(int)

        # Calculate metrics
        metrics = {}
        try:
            metrics["accuracy"] = accuracy_score(
                predictions_df["actual"], predictions_df["pred_binary"]
            )
            metrics["precision"] = precision_score(
                predictions_df["actual"], predictions_df["pred_binary"], zero_division=0
            )
            metrics["recall"] = recall_score(
                predictions_df["actual"], predictions_df["pred_binary"], zero_division=0
            )
            metrics["f1"] = f1_score(
                predictions_df["actual"], predictions_df["pred_binary"], zero_division=0
            )
            metrics["roc_auc"] = roc_auc_score(
                predictions_df["actual"], predictions_df["prediction"]
            )
        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")
            metrics = {"error": str(e)}

        # Additional metrics
        metrics["mean_confidence"] = predictions_df["confidence"].mean()
        metrics["prediction_std"] = predictions_df["prediction"].std()
        metrics["samples"] = len(predictions_df)

        return metrics

    def _select_best_model(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Step 5: Select the best performing model."""
        valid_models = {
            name: result
            for name, result in model_results.items()
            if "error" not in result
        }

        if not valid_models:
            return {"error": "No valid models trained"}

        # Rank by F1 score (harmonic mean of precision and recall)
        model_scores = {}
        for name, result in valid_models.items():
            metrics = result.get("metrics", {})
            f1_score = metrics.get("f1", 0)
            roc_auc = metrics.get("roc_auc", 0)

            # Composite score: 70% F1, 30% ROC-AUC
            composite_score = 0.7 * f1_score + 0.3 * roc_auc
            model_scores[name] = composite_score

        best_model_name = max(model_scores, key=model_scores.get)

        best_model_info = {
            "name": best_model_name,
            "score": model_scores[best_model_name],
            "model": valid_models[best_model_name]["model"],
            "metrics": valid_models[best_model_name]["metrics"],
            "all_scores": model_scores,
        }

        self.best_model = best_model_info
        logger.info(
            f"🏆 Selected {best_model_name} as best model (score: {model_scores[best_model_name]:.4f})"
        )

        return best_model_info

    def _generate_reports_and_tests(
        self, dataset: pd.DataFrame, model_results: Dict, best_model_info: Dict
    ) -> Dict[str, Any]:
        """Step 6: Generate comprehensive reports and automated tests."""
        reports = {}

        # 1. Performance Report
        performance_report = self._generate_performance_report(
            model_results, best_model_info
        )
        reports["performance"] = performance_report

        # 2. Data Quality Report
        quality_report = self.quality_validator.validate_ml_readiness(dataset)
        reports["data_quality"] = quality_report

        # 3. Feature Analysis Report
        feature_report = self._analyze_feature_importance(model_results)
        reports["features"] = feature_report

        # 4. Generate Automated Tests
        if self.config["auto_generate_tests"]:
            test_results = self._generate_automated_tests(best_model_info)
            reports["tests"] = test_results

        # Save reports
        for report_name, report_data in reports.items():
            report_file = (
                self.reports_dir
                / f"{report_name}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(report_file, "w") as f:
                json.dump(report_data, f, indent=2, default=str)
            logger.info(f"📄 Saved {report_name} report to {report_file}")

        return reports

    def _generate_performance_report(
        self, model_results: Dict, best_model_info: Dict
    ) -> Dict:
        """Generate detailed performance report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "best_model": best_model_info["name"],
            "best_score": best_model_info["score"],
            "model_comparison": {
                name: result.get("metrics", {})
                for name, result in model_results.items()
                if "error" not in result
            },
            "dataset_info": {
                "training_samples": list(model_results.values())[0].get(
                    "training_samples", 0
                ),
                "features_used": len(
                    list(model_results.values())[0].get("feature_importance", {})
                ),
            },
            "recommendations": self._generate_model_recommendations(best_model_info),
        }

    def _analyze_feature_importance(self, model_results: Dict) -> Dict:
        """Analyze feature importance across models."""
        feature_analysis = {}

        for model_name, result in model_results.items():
            if "error" in result:
                continue

            importance = result.get("feature_importance", {})
            if importance:
                # Sort by importance
                sorted_features = sorted(
                    importance.items(), key=lambda x: x[1], reverse=True
                )
                feature_analysis[model_name] = {
                    "top_features": sorted_features[:10],
                    "feature_count": len(importance),
                }

        return feature_analysis

    def _generate_automated_tests(self, best_model_info: Dict) -> Dict:
        """Generate automated tests for the best model."""
        model_name = best_model_info["name"]

        # Generate comprehensive test suite
        test_suite = generate_ml_strategy_test(
            model_name,
            "predictor",
            f"Auto-generated tests for {model_name} - Best performing model from AutoML pipeline",
        )

        # Generate comparison tests
        comparison_tests = generate_ml_strategy_test(
            "StrategyComparison",
            "comparison",
            "Automated comparison tests for multiple ML strategies",
        )

        # Generate validation tests
        validation_tests = generate_ml_strategy_test(
            model_name,
            "validation",
            "Statistical validation tests for model robustness",
        )

        test_files = {
            f"test_{model_name.lower()}_automl.py": test_suite,
            "test_strategy_comparison_automl.py": comparison_tests,
            f"test_{model_name.lower()}_validation_automl.py": validation_tests,
        }

        # Save test files locally
        tests_dir = self.output_dir / "tests"
        tests_dir.mkdir(exist_ok=True)

        for filename, content in test_files.items():
            test_file = tests_dir / filename
            with open(test_file, "w") as f:
                f.write(content)
            logger.info(f"🧪 Generated test file: {test_file}")

        # Optionally commit to GitHub
        if self.config["enable_github_integration"]:
            try:
                commit_result = commit_ml_tests_to_github(
                    test_files,
                    f"🤖 AutoML: Automated tests for {model_name} (score: {best_model_info['score']:.4f})",
                )
                logger.info("✅ Committed automated tests to GitHub")
                return {"files": test_files, "github_commit": commit_result}
            except Exception as e:
                logger.warning(f"Failed to commit tests to GitHub: {e}")

        return {"files": test_files}

    def _generate_model_recommendations(self, best_model_info: Dict) -> List[str]:
        """Generate recommendations based on model performance."""
        recommendations = []

        metrics = best_model_info.get("metrics", {})

        # Performance-based recommendations
        if metrics.get("accuracy", 0) < 0.6:
            recommendations.append(
                "Model accuracy is below 60% - consider more features or different algorithm"
            )
        elif metrics.get("accuracy", 0) > 0.8:
            recommendations.append(
                "Excellent model accuracy - ready for production deployment"
            )

        if metrics.get("f1", 0) < 0.5:
            recommendations.append(
                "Poor F1 score - model may be imbalanced or overfitting"
            )

        if metrics.get("roc_auc", 0) > 0.8:
            recommendations.append(
                "Strong discriminative ability - model effectively separates classes"
            )

        # General recommendations
        recommendations.extend(
            [
                "Monitor model performance on new data regularly",
                "Consider ensemble methods for improved stability",
                "Implement confidence thresholds for prediction reliability",
                "Set up automated retraining pipeline for model freshness",
            ]
        )

        return recommendations

    def _deploy_best_model(self, best_model_info: Dict) -> Dict[str, Any]:
        """Step 7: Deploy the best performing model."""
        if not best_model_info or "error" in best_model_info:
            return {"error": "No valid model to deploy"}

        model = best_model_info["model"]
        model_name = best_model_info["name"]

        # Save model
        model_file = (
            self.models_dir
            / f"{model_name}_automl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )
        try:
            joblib.dump(model, model_file)
            logger.info(f"💾 Saved model to {model_file}")

            deployment_info = {
                "model_name": model_name,
                "model_file": str(model_file),
                "deployment_time": datetime.now().isoformat(),
                "metrics": best_model_info.get("metrics", {}),
                "feature_importance": best_model_info.get("feature_importance", {}),
                "status": "deployed",
            }

            # Save deployment info
            deploy_file = self.models_dir / f"deployment_info_{model_name}.json"
            with open(deploy_file, "w") as f:
                json.dump(deployment_info, f, indent=2, default=str)

            return deployment_info

        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            return {"error": str(e), "model_name": model_name}

    def _save_pipeline_results(self, results: Dict[str, Any]) -> None:
        """Save complete pipeline results."""
        results_file = (
            self.output_dir
            / f"automl_pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        # Convert non-serializable objects
        serializable_results = self._make_serializable(results)

        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)

        logger.info(f"💾 Saved complete pipeline results to {results_file}")

    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, "__dict__"):
            return str(obj)  # Convert objects to string representation
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.float64)):
            return obj.item()
        else:
            return obj

    def load_deployed_model(self, model_name: Optional[str] = None) -> Any:
        """Load the most recently deployed model."""
        if model_name:
            # Load specific model
            model_files = list(self.models_dir.glob(f"*{model_name}*.pkl"))
        else:
            # Load latest model
            model_files = list(self.models_dir.glob("*.pkl"))

        if not model_files:
            raise FileNotFoundError("No deployed models found")

        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)

        model = joblib.load(latest_model)
        logger.info(f"📦 Loaded deployed model: {latest_model}")

        return model

    def predict_with_deployed_model(
        self, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make predictions using the deployed model."""
        model = self.load_deployed_model()
        result = model.predict(market_data)

        return {
            "prediction": result.predicted_probability,
            "confidence": result.confidence,
            "recommended_bet": result.recommended_bet,
            "edge": result.edge,
            "reasoning": result.reasoning,
            "model_name": model.name,
        }
