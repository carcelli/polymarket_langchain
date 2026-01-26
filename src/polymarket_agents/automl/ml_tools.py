"""
ML Tools for Agents

LangChain-compatible tools that enable agents to perform machine learning
operations including data ingestion, model training, evaluation, and prediction.
"""

from typing import Dict, List, Any, Optional, Type, Union
from datetime import datetime
import pandas as pd
import json
import sqlite3

from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field

from .data_ingestion import PolymarketDataIngestion
from .data_quality import DataQualityValidator
from .auto_ml_pipeline import AutoMLPipeline
from .ml_database import MLDatabase
from ..ml_strategies import MarketPredictor, EdgeDetector


class DataIngestionInput(BaseModel):
    """Input schema for data ingestion tool."""

    days_back: int = Field(
        description="Number of days of historical data to fetch", default=365
    )
    min_volume: float = Field(
        description="Minimum market volume threshold", default=1000
    )
    include_unresolved: bool = Field(
        description="Whether to include unresolved markets", default=False
    )


class DataQualityCheckInput(BaseModel):
    """Input schema for data quality check tool."""

    dataset_info: str = Field(description="Description of the dataset to check")


class RunAutoMLPipelineInput(BaseModel):
    """Input schema for AutoML pipeline tool."""

    experiment_name: str = Field(description="Name for the ML experiment")
    days_back: int = Field(description="Days of historical data to use", default=365)
    min_volume: float = Field(description="Minimum volume threshold", default=1000)
    models_to_train: List[str] = Field(
        description="Models to train", default=["MarketPredictor", "EdgeDetector"]
    )


class TrainMLModelInput(BaseModel):
    """Input schema for model training tool."""

    model_type: str = Field(
        description="Type of model to train (MarketPredictor, EdgeDetector)"
    )
    experiment_name: str = Field(description="Experiment name for tracking")
    hyperparameters: Optional[Dict[str, Any]] = Field(
        description="Model hyperparameters", default=None
    )


class EvaluateModelInput(BaseModel):
    """Input schema for model evaluation tool."""

    model_id: str = Field(description="ID of the model to evaluate")
    evaluation_type: str = Field(
        description="Type of evaluation (backtest, cross_validation)",
        default="backtest",
    )


class MakePredictionInput(BaseModel):
    """Input schema for prediction tool."""

    model_id: str = Field(description="ID of the model to use for prediction")
    market_data: Dict[str, Any] = Field(description="Market data for prediction")


class GetExperimentResultsInput(BaseModel):
    """Input schema for getting experiment results."""

    experiment_id: str = Field(description="ID of the experiment to retrieve")


class DataIngestionTool(BaseTool):
    """Tool for ingesting and preparing ML data from Polymarket."""

    name: str = "data_ingestion"
    description: str = (
        "Ingest historical market data from Polymarket and prepare it for ML training"
    )
    args_schema: Type[BaseModel] = DataIngestionInput

    def _run(
        self,
        days_back: int = 365,
        min_volume: float = 1000,
        include_unresolved: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run data ingestion pipeline."""
        try:
            ingestion = PolymarketDataIngestion()

            # Run data pipeline
            dataset = ingestion.run_data_pipeline(days_back=days_back)

            # Apply filters
            dataset = dataset[dataset["volume"] >= min_volume]
            if not include_unresolved:
                dataset = dataset[dataset["resolved"]]

            # Store in database
            db = MLDatabase()
            experiment_id = db.create_experiment(
                name=f"Data Ingestion {datetime.now().strftime('%Y%m%d')}",
                description=f"Ingested {days_back} days of market data",
            )

            # Save dataset metadata
            dataset_id = db.save_dataset(
                experiment_id=experiment_id,
                name="ingested_market_data",
                dataset_type="full",
                sample_count=len(dataset),
                feature_count=len(dataset.columns) - 1,  # Exclude target
                target_distribution=dataset["will_resolve_yes"]
                .value_counts()
                .to_dict(),
            )

            db.update_experiment_status(experiment_id, "completed", success=True)

            return json.dumps(
                {
                    "status": "success",
                    "samples_ingested": len(dataset),
                    "features_created": len(dataset.columns),
                    "experiment_id": experiment_id,
                    "dataset_id": dataset_id,
                    "target_distribution": dataset["will_resolve_yes"]
                    .value_counts()
                    .to_dict(),
                },
                indent=2,
            )

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)}, indent=2)


class DataQualityTool(BaseTool):
    """Tool for validating data quality for ML."""

    name: str = "data_quality_check"
    description: str = "Validate data quality and readiness for ML training"
    args_schema: Type[BaseModel] = DataQualityCheckInput

    def _run(
        self, dataset_info: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Run data quality validation."""
        try:
            # Load recent dataset (simplified - in practice, would load specific dataset)
            ingestion = PolymarketDataIngestion()
            dataset = ingestion.create_training_dataset(days_back=180, min_volume=1000)

            if dataset.empty:
                return json.dumps(
                    {
                        "status": "error",
                        "error": "No dataset available for quality check",
                    }
                )

            # Run quality validation
            validator = DataQualityValidator()
            quality_report = validator.validate_ml_readiness(dataset)

            # Store results
            db = MLDatabase()
            experiment_id = db.create_experiment(
                name=f"Data Quality Check {datetime.now().strftime('%Y%m%d')}",
                description=f"Quality validation for {dataset_info}",
            )

            db.update_experiment_status(experiment_id, "completed", success=True)

            return json.dumps(
                {
                    "status": "success",
                    "readiness_score": quality_report["readiness_score"],
                    "ready_for_ml": quality_report["ready_for_ml"],
                    "issues_count": len(quality_report["quality_check"]["issues"]),
                    "recommendations": quality_report.get("recommendations", [])[:5],
                    "experiment_id": experiment_id,
                },
                indent=2,
            )

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)}, indent=2)


class AutoMLPipelineTool(BaseTool):
    """Tool for running complete AutoML pipelines."""

    name: str = "run_automl_pipeline"
    description: str = "Run complete automated ML pipeline from data to deployed model"
    args_schema: Type[BaseModel] = RunAutoMLPipelineInput

    def _run(
        self,
        experiment_name: str,
        days_back: int = 365,
        min_volume: float = 1000,
        models_to_train: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run complete AutoML pipeline."""
        if models_to_train is None:
            models_to_train = ["MarketPredictor", "EdgeDetector"]

        try:
            # Configure pipeline
            config = {
                "output_dir": f'./automl_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                "data_days_back": days_back,
                "min_volume": min_volume,
                "models_to_train": models_to_train,
                "enable_github_integration": False,
                "auto_generate_tests": False,
            }

            # Run pipeline
            pipeline = AutoMLPipeline(config)
            results = pipeline.run_pipeline()

            if results["success"]:
                return json.dumps(
                    {
                        "status": "success",
                        "experiment_name": experiment_name,
                        "best_model": results["best_model"]["name"],
                        "best_score": results["best_model"]["score"],
                        "data_samples": results["data_summary"]["final_samples"],
                        "features_used": results["data_summary"]["features_count"],
                        "pipeline_duration": results["pipeline_metadata"][
                            "duration_seconds"
                        ],
                        "model_metrics": results["best_model"]["metrics"],
                    },
                    indent=2,
                )
            else:
                return json.dumps(
                    {
                        "status": "error",
                        "error": results.get("error", "Unknown pipeline error"),
                    },
                    indent=2,
                )

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)}, indent=2)


class ModelTrainingTool(BaseTool):
    """Tool for training individual ML models."""

    name: str = "train_ml_model"
    description: str = "Train a specific ML model for market prediction"
    args_schema: Type[BaseModel] = TrainMLModelInput

    def _run(
        self,
        model_type: str,
        experiment_name: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Train a specific ML model."""
        try:
            # Initialize database and create experiment
            db = MLDatabase()
            experiment_id = db.create_experiment(
                name=f"{experiment_name} - {model_type}",
                description=f"Training {model_type} model",
            )

            # Get training data
            ingestion = PolymarketDataIngestion()
            dataset = ingestion.create_training_dataset(days_back=365, min_volume=1000)

            if dataset.empty:
                return json.dumps(
                    {"status": "error", "error": "No training data available"}
                )

            # Initialize model
            model: Union[MarketPredictor, EdgeDetector]
            if model_type == "MarketPredictor":
                model = MarketPredictor()
            elif model_type == "EdgeDetector":
                model = EdgeDetector()
            else:
                return json.dumps(
                    {"status": "error", "error": f"Unknown model type: {model_type}"}
                )

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

            # Save model to database
            model_info = {
                "name": f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "model_type": model_type,
                "algorithm": model_type,  # Simplified
                "hyperparameters": hyperparameters or {},
                "feature_columns": (
                    list(model.feature_columns)
                    if hasattr(model, "feature_columns")
                    else []
                ),
                "training_samples": len(train_data),
                "training_start_time": datetime.now().isoformat(),
                "training_end_time": datetime.now().isoformat(),
            }

            model_id = db.save_model(experiment_id, model_info)

            # Evaluate model
            predictions = []
            for _, row in dataset.iterrows():
                result = model.predict(row.to_dict())
                predictions.append(
                    {
                        "market_id": result.market_id,
                        "predicted_probability": result.predicted_probability,
                        "actual_outcome": row.get("will_resolve_yes"),
                        "confidence": result.confidence,
                        "recommended_bet": result.recommended_bet,
                    }
                )

            # Calculate metrics
            pred_df = pd.DataFrame(predictions)
            metrics = self._calculate_metrics(pred_df)

            # Save metrics and predictions
            db.save_model_metrics(model_id, metrics)
            db.save_predictions(model_id, predictions)

            db.update_experiment_status(experiment_id, "completed", success=True)

            return json.dumps(
                {
                    "status": "success",
                    "model_id": model_id,
                    "model_type": model_type,
                    "training_samples": len(train_data),
                    "metrics": metrics,
                    "experiment_id": experiment_id,
                },
                indent=2,
            )

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)}, indent=2)

    def _calculate_metrics(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate model performance metrics."""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
        )

        # Filter out rows without actual outcomes
        valid_predictions = predictions_df.dropna(subset=["actual_outcome"])

        if len(valid_predictions) == 0:
            return {"error": "No valid predictions with actual outcomes"}

        # Binary predictions
        valid_predictions["pred_binary"] = (
            valid_predictions["predicted_probability"] > 0.5
        ).astype(int)

        try:
            metrics = {
                "accuracy": accuracy_score(
                    valid_predictions["actual_outcome"],
                    valid_predictions["pred_binary"],
                ),
                "precision": precision_score(
                    valid_predictions["actual_outcome"],
                    valid_predictions["pred_binary"],
                    zero_division=0,
                ),
                "recall": recall_score(
                    valid_predictions["actual_outcome"],
                    valid_predictions["pred_binary"],
                    zero_division=0,
                ),
                "f1": f1_score(
                    valid_predictions["actual_outcome"],
                    valid_predictions["pred_binary"],
                    zero_division=0,
                ),
                "roc_auc": roc_auc_score(
                    valid_predictions["actual_outcome"],
                    valid_predictions["predicted_probability"],
                ),
                "samples": len(valid_predictions),
            }
        except Exception as e:
            metrics = {"error": str(e)}

        return metrics


class ModelEvaluationTool(BaseTool):
    """Tool for evaluating trained ML models."""

    name: str = "evaluate_ml_model"
    description: str = "Evaluate a trained ML model's performance"
    args_schema: Type[BaseModel] = EvaluateModelInput

    def _run(
        self,
        model_id: str,
        evaluation_type: str = "backtest",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Evaluate a trained model."""
        try:
            db = MLDatabase()

            # Get model info
            with db._ensure_database():  # Ensure connection
                conn = sqlite3.connect(db.db_path)
                model_info = conn.execute(
                    "SELECT * FROM models WHERE model_id = ?", (model_id,)
                ).fetchone()

                if not model_info:
                    return json.dumps(
                        {"status": "error", "error": f"Model {model_id} not found"}
                    )

                model_dict = dict(
                    zip([desc[0] for desc in model_info.description], model_info)
                )

            # Load model (simplified - in practice, load from model_path)
            model_type = model_dict["model_type"]
            model: Union[MarketPredictor, EdgeDetector]
            if model_type == "MarketPredictor":
                model = MarketPredictor()
            elif model_type == "EdgeDetector":
                model = EdgeDetector()
            else:
                return json.dumps(
                    {"status": "error", "error": f"Unknown model type: {model_type}"}
                )

            # Get test data
            ingestion = PolymarketDataIngestion()
            test_data = ingestion.create_training_dataset(
                days_back=180, min_volume=2000
            )

            if test_data.empty:
                return json.dumps(
                    {"status": "error", "error": "No test data available"}
                )

            # Run evaluation
            evaluation_config = {
                "evaluation_type": evaluation_type,
                "test_samples": len(test_data),
                "model_type": model_type,
            }

            results = {}
            start_time = datetime.now()

            if evaluation_type == "backtest":
                results = self._run_backtest_evaluation(model, test_data)
            elif evaluation_type == "cross_validation":
                results = self._run_cross_validation(model, test_data)
            else:
                results = {"error": f"Unknown evaluation type: {evaluation_type}"}

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Save evaluation results
            db.save_evaluation(
                model_id, evaluation_type, evaluation_config, results, duration
            )

            return json.dumps(
                {
                    "status": "success",
                    "model_id": model_id,
                    "evaluation_type": evaluation_type,
                    "results": results,
                    "duration_seconds": duration,
                },
                indent=2,
            )

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)}, indent=2)

    def _run_backtest_evaluation(
        self, model, test_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run backtesting evaluation."""
        predictions = []

        for _, row in test_data.iterrows():
            result = model.predict(row.to_dict())
            predictions.append(
                {
                    "market_id": result.market_id,
                    "predicted_probability": result.predicted_probability,
                    "actual_outcome": row.get("will_resolve_yes"),
                    "recommended_bet": result.recommended_bet,
                    "position_size": result.position_size,
                    "expected_value": result.expected_value,
                }
            )

        pred_df = pd.DataFrame(predictions)

        # Calculate trading metrics
        valid_trades = pred_df[pred_df["recommended_bet"] != "PASS"]

        if len(valid_trades) > 0:
            win_rate = (valid_trades["predicted_probability"] > 0.5).mean()
            avg_position = valid_trades["position_size"].mean()
            total_trades = len(valid_trades)
        else:
            win_rate = 0
            avg_position = 0
            total_trades = 0

        return {
            "total_predictions": len(predictions),
            "valid_trades": total_trades,
            "win_rate": win_rate,
            "average_position_size": avg_position,
            "total_markets": len(test_data),
        }

    def _run_cross_validation(self, model, data: pd.DataFrame) -> Dict[str, Any]:
        """Run cross-validation evaluation."""
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier

        # Prepare data for sklearn
        feature_cols = [
            col
            for col in data.columns
            if col not in ["will_resolve_yes", "market_id", "question"]
        ]
        X = data[feature_cols].fillna(0)
        y = data["will_resolve_yes"]

        # Simple cross-validation with RandomForest (placeholder)
        clf = RandomForestClassifier(n_estimators=10, random_state=42)

        try:
            cv_scores = cross_val_score(clf, X, y, cv=3, scoring="f1")
            return {
                "cv_f1_scores": cv_scores.tolist(),
                "cv_f1_mean": cv_scores.mean(),
                "cv_f1_std": cv_scores.std(),
                "cv_folds": len(cv_scores),
            }
        except Exception as e:
            return {"error": str(e)}


class PredictionTool(BaseTool):
    """Tool for making predictions with trained models."""

    name: str = "make_ml_prediction"
    description: str = "Make predictions on market data using a trained ML model"
    args_schema: Type[BaseModel] = MakePredictionInput

    def _run(
        self,
        model_id: str,
        market_data: Dict[str, Any],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Make prediction with trained model."""
        try:
            db = MLDatabase()

            # Get model info (simplified - in practice load actual model)
            # For now, create a fresh model of the right type
            conn = sqlite3.connect(db.db_path)
            model_info = conn.execute(
                "SELECT model_type FROM models WHERE model_id = ?", (model_id,)
            ).fetchone()

            if not model_info:
                return json.dumps(
                    {"status": "error", "error": f"Model {model_id} not found"}
                )

            model_type = model_info[0]

            model: Union[MarketPredictor, EdgeDetector]
            if model_type == "MarketPredictor":
                model = MarketPredictor()
            elif model_type == "EdgeDetector":
                model = EdgeDetector()
            else:
                return json.dumps(
                    {"status": "error", "error": f"Unknown model type: {model_type}"}
                )

            # Make prediction
            result = model.predict(market_data)

            # Save prediction to database
            prediction_data = {
                "market_id": result.market_id,
                "predicted_probability": result.predicted_probability,
                "confidence": result.confidence,
                "recommended_bet": result.recommended_bet,
                "position_size": result.position_size,
                "expected_value": result.expected_value,
                "market_data": market_data,
            }

            db.save_predictions(model_id, [prediction_data])

            return json.dumps(
                {
                    "status": "success",
                    "model_id": model_id,
                    "market_id": result.market_id,
                    "predicted_probability": result.predicted_probability,
                    "confidence": result.confidence,
                    "recommended_bet": result.recommended_bet,
                    "position_size": result.position_size,
                    "expected_value": result.expected_value,
                    "reasoning": result.reasoning,
                },
                indent=2,
            )

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)}, indent=2)


class ExperimentResultsTool(BaseTool):
    """Tool for retrieving experiment results and model performance."""

    name: str = "get_experiment_results"
    description: str = "Retrieve results and performance metrics for ML experiments"
    args_schema: Type[BaseModel] = GetExperimentResultsInput

    def _run(
        self,
        experiment_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get experiment results."""
        try:
            db = MLDatabase()
            results = db.get_experiment_results(experiment_id)

            if not results:
                return json.dumps(
                    {
                        "status": "error",
                        "error": f"Experiment {experiment_id} not found",
                    }
                )

            # Format results for agent consumption
            formatted_results = {
                "experiment_id": experiment_id,
                "experiment_name": results["experiment"]["name"],
                "status": results["experiment"]["status"],
                "success": results["experiment"]["success"],
                "duration_seconds": results["experiment"]["duration_seconds"],
                "models_trained": len(results["models"]),
                "datasets_created": len(results["datasets"]),
            }

            # Add best model info
            if results["models"]:
                best_model = max(
                    results["models"],
                    key=lambda m: (
                        m.get("metrics", [{}])[0].get("value", 0)
                        if m.get("metrics")
                        else 0
                    ),
                )
                formatted_results["best_model"] = {
                    "name": best_model["name"],
                    "type": best_model["model_type"],
                    "metrics": best_model.get("metrics", []),
                }

            return json.dumps(formatted_results, indent=2)

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)}, indent=2)


# Create tool instances for easy import
data_ingestion_tool = DataIngestionTool()
data_quality_tool = DataQualityTool()
automl_pipeline_tool = AutoMLPipelineTool()
model_training_tool = ModelTrainingTool()
model_evaluation_tool = ModelEvaluationTool()
prediction_tool = PredictionTool()
experiment_results_tool = ExperimentResultsTool()

# List of all ML tools
ml_tools = [
    data_ingestion_tool,
    data_quality_tool,
    automl_pipeline_tool,
    model_training_tool,
    model_evaluation_tool,
    prediction_tool,
    experiment_results_tool,
]
