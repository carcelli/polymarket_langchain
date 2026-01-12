"""
ML Database Schema and Operations

Comprehensive database system for storing machine learning experiments,
models, predictions, and performance metrics in an organized structure.
"""

import sqlite3
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MLDatabase:
    """
    Database system for ML experiments, models, and results.

    Provides structured storage for:
    - Experiments (pipeline runs)
    - Models (trained models with metadata)
    - Predictions (model outputs)
    - Evaluations (performance metrics)
    - Features (feature engineering results)
    - Datasets (data versions and splits)
    """

    def __init__(self, db_path: str = "data/ml_experiments.db"):
        self.db_path = db_path
        self._ensure_database()

    def _ensure_database(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT DEFAULT 'running',  -- running, completed, failed
                    pipeline_config TEXT,  -- JSON config
                    start_time TEXT,
                    end_time TEXT,
                    duration_seconds REAL,
                    success BOOLEAN,
                    error_message TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    name TEXT NOT NULL,
                    model_type TEXT,  -- MarketPredictor, EdgeDetector, etc.
                    algorithm TEXT,   -- RandomForest, NeuralNetwork, etc.
                    hyperparameters TEXT,  -- JSON hyperparameters
                    feature_columns TEXT,  -- JSON list of features used
                    training_samples INTEGER,
                    training_start_time TEXT,
                    training_end_time TEXT,
                    model_path TEXT,  -- Path to serialized model
                    model_size_bytes INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_metrics (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT,
                    metric_name TEXT,  -- accuracy, f1, auc, etc.
                    metric_value REAL,
                    dataset_type TEXT,  -- train, validation, test
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES models(model_id)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT,
                    market_id TEXT,
                    predicted_probability REAL,
                    actual_outcome REAL,  -- NULL if not resolved
                    confidence REAL,
                    recommended_bet TEXT,  -- YES, NO, PASS
                    position_size REAL,
                    expected_value REAL,
                    prediction_time TEXT,
                    market_data TEXT,  -- JSON market data snapshot
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES models(model_id)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS evaluations (
                    evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT,
                    evaluation_type TEXT,  -- backtest, cross_validation, etc.
                    evaluation_config TEXT,  -- JSON config
                    results TEXT,  -- JSON results
                    start_time TEXT,
                    end_time TEXT,
                    duration_seconds REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES models(model_id)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS feature_sets (
                    feature_set_id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    name TEXT,
                    feature_columns TEXT,  -- JSON list of features
                    feature_stats TEXT,    -- JSON statistics
                    sample_count INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    name TEXT,
                    dataset_type TEXT,  -- train, validation, test, full
                    sample_count INTEGER,
                    feature_count INTEGER,
                    target_distribution TEXT,  -- JSON class distribution
                    data_path TEXT,  -- Path to stored dataset
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS ml_alerts (
                    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT,
                    alert_type TEXT,  -- performance_drop, drift_detected, etc.
                    severity TEXT,    -- low, medium, high, critical
                    message TEXT,
                    details TEXT,     -- JSON additional details
                    resolved BOOLEAN DEFAULT FALSE,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TEXT,
                    FOREIGN KEY (model_id) REFERENCES models(model_id)
                )
            ''')

    def create_experiment(self, name: str, description: str = "",
                         pipeline_config: Dict[str, Any] = None) -> str:
        """
        Create a new ML experiment.

        Args:
            name: Experiment name
            description: Experiment description
            pipeline_config: Pipeline configuration

        Returns:
            Experiment ID
        """
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name.replace(' ', '_').lower()}"

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO experiments
                (experiment_id, name, description, pipeline_config, start_time)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                experiment_id,
                name,
                description,
                json.dumps(pipeline_config or {}),
                datetime.now().isoformat()
            ))

        logger.info(f"Created experiment: {experiment_id}")
        return experiment_id

    def update_experiment_status(self, experiment_id: str, status: str,
                               success: bool = None, error_message: str = None):
        """
        Update experiment status.

        Args:
            experiment_id: Experiment ID
            status: New status (running, completed, failed)
            success: Whether experiment succeeded
            error_message: Error message if failed
        """
        end_time = datetime.now().isoformat() if status in ['completed', 'failed'] else None

        with sqlite3.connect(self.db_path) as conn:
            # Get start time to calculate duration
            start_time = conn.execute(
                "SELECT start_time FROM experiments WHERE experiment_id = ?",
                (experiment_id,)
            ).fetchone()

            duration = None
            if start_time and end_time:
                start_dt = pd.to_datetime(start_time[0])
                end_dt = pd.to_datetime(end_time)
                duration = (end_dt - start_dt).total_seconds()

            conn.execute('''
                UPDATE experiments
                SET status = ?, end_time = ?, duration_seconds = ?,
                    success = ?, error_message = ?
                WHERE experiment_id = ?
            ''', (status, end_time, duration, success, error_message, experiment_id))

        logger.info(f"Updated experiment {experiment_id} status to {status}")

    def save_model(self, experiment_id: str, model_info: Dict[str, Any]) -> str:
        """
        Save model metadata to database.

        Args:
            experiment_id: Parent experiment ID
            model_info: Model information dictionary

        Returns:
            Model ID
        """
        model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_info['name'].replace(' ', '_').lower()}"

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO models
                (model_id, experiment_id, name, model_type, algorithm,
                 hyperparameters, feature_columns, training_samples,
                 training_start_time, training_end_time, model_path, model_size_bytes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_id,
                experiment_id,
                model_info['name'],
                model_info.get('model_type', 'unknown'),
                model_info.get('algorithm', 'unknown'),
                json.dumps(model_info.get('hyperparameters', {})),
                json.dumps(model_info.get('feature_columns', [])),
                model_info.get('training_samples', 0),
                model_info.get('training_start_time'),
                model_info.get('training_end_time'),
                model_info.get('model_path'),
                model_info.get('model_size_bytes', 0)
            ))

        logger.info(f"Saved model: {model_id}")
        return model_id

    def save_model_metrics(self, model_id: str, metrics: Dict[str, float],
                          dataset_type: str = 'test'):
        """
        Save model performance metrics.

        Args:
            model_id: Model ID
            metrics: Dictionary of metric names -> values
            dataset_type: Dataset type (train, validation, test)
        """
        with sqlite3.connect(self.db_path) as conn:
            for metric_name, metric_value in metrics.items():
                conn.execute('''
                    INSERT INTO model_metrics
                    (model_id, metric_name, metric_value, dataset_type)
                    VALUES (?, ?, ?, ?)
                ''', (model_id, metric_name, metric_value, dataset_type))

        logger.info(f"Saved {len(metrics)} metrics for model {model_id}")

    def save_predictions(self, model_id: str, predictions: List[Dict[str, Any]]):
        """
        Save model predictions.

        Args:
            model_id: Model ID
            predictions: List of prediction dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            for pred in predictions:
                conn.execute('''
                    INSERT INTO predictions
                    (model_id, market_id, predicted_probability, actual_outcome,
                     confidence, recommended_bet, position_size, expected_value,
                     prediction_time, market_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    model_id,
                    pred['market_id'],
                    pred.get('predicted_probability'),
                    pred.get('actual_outcome'),
                    pred.get('confidence'),
                    pred.get('recommended_bet'),
                    pred.get('position_size', 0),
                    pred.get('expected_value', 0),
                    pred.get('prediction_time', datetime.now().isoformat()),
                    json.dumps(pred.get('market_data', {}))
                ))

        logger.info(f"Saved {len(predictions)} predictions for model {model_id}")

    def save_evaluation(self, model_id: str, evaluation_type: str,
                       evaluation_config: Dict[str, Any], results: Dict[str, Any],
                       duration_seconds: float = None):
        """
        Save model evaluation results.

        Args:
            model_id: Model ID
            evaluation_type: Type of evaluation (backtest, cross_validation, etc.)
            evaluation_config: Evaluation configuration
            results: Evaluation results
            duration_seconds: Evaluation duration
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO evaluations
                (model_id, evaluation_type, evaluation_config, results,
                 start_time, end_time, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_id,
                evaluation_type,
                json.dumps(evaluation_config),
                json.dumps(results),
                datetime.now().isoformat(),  # start_time (simplified)
                datetime.now().isoformat(),  # end_time (simplified)
                duration_seconds
            ))

        logger.info(f"Saved {evaluation_type} evaluation for model {model_id}")

    def save_feature_set(self, experiment_id: str, name: str,
                        feature_columns: List[str], feature_stats: Dict[str, Any] = None,
                        sample_count: int = 0) -> str:
        """
        Save feature set metadata.

        Args:
            experiment_id: Parent experiment ID
            name: Feature set name
            feature_columns: List of feature column names
            feature_stats: Feature statistics
            sample_count: Number of samples

        Returns:
            Feature set ID
        """
        feature_set_id = f"feat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name.replace(' ', '_').lower()}"

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO feature_sets
                (feature_set_id, experiment_id, name, feature_columns,
                 feature_stats, sample_count)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                feature_set_id,
                experiment_id,
                name,
                json.dumps(feature_columns),
                json.dumps(feature_stats or {}),
                sample_count
            ))

        logger.info(f"Saved feature set: {feature_set_id}")
        return feature_set_id

    def save_dataset(self, experiment_id: str, name: str, dataset_type: str,
                    sample_count: int, feature_count: int,
                    target_distribution: Dict[str, int] = None,
                    data_path: str = None) -> str:
        """
        Save dataset metadata.

        Args:
            experiment_id: Parent experiment ID
            name: Dataset name
            dataset_type: Type (train, validation, test, full)
            sample_count: Number of samples
            feature_count: Number of features
            target_distribution: Class distribution
            data_path: Path to stored dataset

        Returns:
            Dataset ID
        """
        dataset_id = f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name.replace(' ', '_').lower()}"

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO datasets
                (dataset_id, experiment_id, name, dataset_type, sample_count,
                 feature_count, target_distribution, data_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                dataset_id,
                experiment_id,
                name,
                dataset_type,
                sample_count,
                feature_count,
                json.dumps(target_distribution or {}),
                data_path
            ))

        logger.info(f"Saved dataset: {dataset_id}")
        return dataset_id

    def create_alert(self, model_id: str, alert_type: str, severity: str,
                    message: str, details: Dict[str, Any] = None):
        """
        Create a new ML alert.

        Args:
            model_id: Model ID
            alert_type: Type of alert (performance_drop, drift_detected, etc.)
            severity: Alert severity (low, medium, high, critical)
            message: Alert message
            details: Additional alert details
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO ml_alerts
                (model_id, alert_type, severity, message, details)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                model_id,
                alert_type,
                severity,
                message,
                json.dumps(details or {})
            ))

        logger.info(f"Created {severity} alert for model {model_id}: {message}")

    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get comprehensive results for an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment results dictionary
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get experiment info
            exp_query = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,)
            ).fetchone()

            if not exp_query:
                return None

            experiment_info = dict(zip([desc[0] for desc in exp_query.description], exp_query))

            # Get models
            models_query = conn.execute(
                "SELECT * FROM models WHERE experiment_id = ?",
                (experiment_id,)
            ).fetchall()

            models = []
            for model_row in models_query:
                model_dict = dict(zip([desc[0] for desc in model_row.description], model_row))

                # Get metrics for this model
                metrics_query = conn.execute(
                    "SELECT metric_name, metric_value, dataset_type FROM model_metrics WHERE model_id = ?",
                    (model_dict['model_id'],)
                ).fetchall()

                model_dict['metrics'] = [
                    {
                        'name': row[0],
                        'value': row[1],
                        'dataset_type': row[2]
                    } for row in metrics_query
                ]

                models.append(model_dict)

            # Get datasets
            datasets_query = conn.execute(
                "SELECT * FROM datasets WHERE experiment_id = ?",
                (experiment_id,)
            ).fetchall()

            datasets = [dict(zip([desc[0] for desc in dataset.description], dataset))
                       for dataset in datasets_query]

            return {
                'experiment': experiment_info,
                'models': models,
                'datasets': datasets
            }

    def get_model_performance_history(self, model_id: str) -> pd.DataFrame:
        """
        Get performance history for a model.

        Args:
            model_id: Model ID

        Returns:
            DataFrame with performance metrics over time
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query('''
                SELECT metric_name, metric_value, dataset_type, created_at
                FROM model_metrics
                WHERE model_id = ?
                ORDER BY created_at
            ''', conn, params=(model_id,))

        return df

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """
        Get all unresolved alerts.

        Returns:
            List of active alert dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            alerts_query = conn.execute(
                "SELECT * FROM ml_alerts WHERE resolved = FALSE ORDER BY created_at DESC"
            ).fetchall()

            alerts = []
            for alert_row in alerts_query:
                alert_dict = dict(zip([desc[0] for desc in alert_row.description], alert_row))
                alerts.append(alert_dict)

            return alerts

    def get_best_models(self, limit: int = 10, metric: str = 'f1') -> pd.DataFrame:
        """
        Get best performing models by a specific metric.

        Args:
            limit: Number of models to return
            metric: Metric to rank by

        Returns:
            DataFrame with best models
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f'''
                SELECT m.model_id, m.name, m.model_type, m.algorithm,
                       m.training_samples, m.created_at,
                       mm.metric_value as {metric}_score
                FROM models m
                JOIN model_metrics mm ON m.model_id = mm.model_id
                WHERE mm.metric_name = ?
                ORDER BY mm.metric_value DESC
                LIMIT ?
            ''', conn, params=(metric, limit))

        return df

    def export_experiment_data(self, experiment_id: str, output_path: str):
        """
        Export all data for an experiment to JSON.

        Args:
            experiment_id: Experiment ID
            output_path: Path to save exported data
        """
        results = self.get_experiment_results(experiment_id)

        if results:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Exported experiment {experiment_id} to {output_path}")
        else:
            logger.warning(f"No data found for experiment {experiment_id}")

    def cleanup_old_experiments(self, days_to_keep: int = 30):
        """
        Clean up old experiment data.

        Args:
            days_to_keep: Number of days of data to keep
        """
        cutoff_date = (datetime.now() - pd.Timedelta(days=days_to_keep)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # This is a simplified cleanup - in practice, you'd want more sophisticated archiving
            deleted_count = conn.execute(
                "DELETE FROM experiments WHERE created_at < ?",
                (cutoff_date,)
            ).rowcount

        logger.info(f"Cleaned up {deleted_count} old experiments")

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive database statistics.

        Returns:
            Database statistics dictionary
        """
        with sqlite3.connect(self.db_path) as conn:
            stats = {}

            # Count records in each table
            tables = ['experiments', 'models', 'model_metrics', 'predictions',
                     'evaluations', 'feature_sets', 'datasets', 'ml_alerts']

            for table in tables:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                stats[f'{table}_count'] = count

            # Get database file size
            db_size = Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
            stats['database_size_bytes'] = db_size

            # Get recent activity
            recent_experiments = conn.execute(
                "SELECT COUNT(*) FROM experiments WHERE created_at >= datetime('now', '-7 days')"
            ).fetchone()[0]
            stats['experiments_last_7_days'] = recent_experiments

            return stats
