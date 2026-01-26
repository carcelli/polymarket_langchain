"""
XGBoost Probability Calibration Strategy

Uses gradient boosting to predict the probability that a market will resolve YES,
providing calibrated probability estimates for edge detection and betting decisions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging

from polymarket_agents.ml_strategies.base_strategy import (
    MLBettingStrategy,
    StrategyResult,
)
from polymarket_agents.ml_strategies.registry import register_strategy
from polymarket_agents.automl.data_ingestion import PolymarketDataIngestion

# Try to import XGBoost, fall back to sklearn if not available
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    try:
        from sklearn.ensemble import GradientBoostingClassifier

        SKLEARN_AVAILABLE = True
    except ImportError:
        SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


@register_strategy("xgboost_probability")
class XGBoostProbabilityStrategy(MLBettingStrategy):
    """
    Gradient Boosting-based strategy for probability calibration and edge detection.

    Uses XGBoost if available, otherwise falls back to sklearn's GradientBoostingClassifier.
    Trains a gradient boosting model to predict the true probability that a market
    will resolve YES, then compares this to the current market-implied probability
    to identify edges and betting opportunities.
    """

    def __init__(
        self,
        name: str = "gradient_boosting_probability",
        model_path: str = "data/models/gradient_boosting_model.pkl",
    ):
        super().__init__(name)
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model: Optional[Any] = None
        self.feature_names: Optional[List[str]] = None
        self.use_xgboost = XGBOOST_AVAILABLE
        self.data_ingestor = PolymarketDataIngestion()

        if not (XGBOOST_AVAILABLE or SKLEARN_AVAILABLE):
            raise ImportError(
                "Neither XGBoost nor scikit-learn available. Install with: pip install xgboost or pip install scikit-learn"
            )

    def _prepare_features_for_training(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and target for model training.

        Args:
            df: DataFrame with market data and engineered features

        Returns:
            Tuple of (X, y, feature_names)
        """
        # Define feature columns (exclude target, metadata, and leaky features)
        exclude_cols = [
            "market_id",
            "question",
            "description",
            "outcomes",
            "created_at",
            "end_date",
            "resolved_at",
            "actual_outcome",
            "will_resolve_yes",
            "target",
            "resolved",
            "active",
            "category",  # Keep category for one-hot encoding
            # Exclude leaky features that wouldn't be available during prediction
            "true_prob",
            "edge",
            "abs_edge",
        ]

        # Get numeric and boolean columns
        feature_cols = []
        for col in df.columns:
            if col not in exclude_cols:
                if df[col].dtype in ["int64", "float64", "bool"]:
                    feature_cols.append(col)
                elif df[col].dtype == "object" and col == "volume_category":
                    # Handle categorical volume
                    feature_cols.append(col)

        logger.info(f"Using {len(feature_cols)} features: {feature_cols}")

        # Prepare X
        X = df[feature_cols].copy()

        # Handle categorical variables
        if "volume_category" in X.columns:
            # One-hot encode volume categories
            volume_dummies = pd.get_dummies(X["volume_category"], prefix="volume_cat")
            X = pd.concat([X.drop("volume_category", axis=1), volume_dummies], axis=1)

        # Handle any remaining NaN values
        X = X.fillna(0)

        # Prepare target - try different column names
        target_col = None
        for col in ["will_resolve_yes", "target", "actual_outcome"]:
            if col in df.columns:
                target_col = col
                break

        if target_col is None:
            raise ValueError(
                "No target column found. Expected 'will_resolve_yes', 'target', or 'actual_outcome'"
            )

        y = df[target_col].astype(int).values
        logger.info(f"Using target column: {target_col}")

        # Get final feature names
        feature_names = list(X.columns)

        return X.values, y, feature_names

    def _prepare_features_for_prediction(
        self, market_data: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features for prediction (without target variable).

        Args:
            market_data: Dictionary containing market information

        Returns:
            Tuple of (X, feature_names)
        """
        # Convert to DataFrame for processing
        df = pd.DataFrame([market_data])

        # Define feature columns (same as training, exclude target and metadata)
        exclude_cols = [
            "market_id",
            "question",
            "description",
            "outcomes",
            "created_at",
            "end_date",
            "resolved_at",
            "actual_outcome",
            "will_resolve_yes",
            "resolved",
            "active",
            "category",  # Keep category for one-hot encoding
        ]

        # Get numeric and boolean columns
        feature_cols = []
        for col in df.columns:
            if col not in exclude_cols:
                if df[col].dtype in ["int64", "float64", "bool"]:
                    feature_cols.append(col)
                elif df[col].dtype == "object" and col == "volume_category":
                    # Handle categorical volume
                    feature_cols.append(col)

        # Prepare X
        X = df[feature_cols].copy()

        # Handle categorical variables
        if "volume_category" in X.columns:
            # One-hot encode volume categories
            volume_dummies = pd.get_dummies(X["volume_category"], prefix="volume_cat")
            X = pd.concat([X.drop("volume_category", axis=1), volume_dummies], axis=1)

        # Handle any remaining NaN values
        X = X.fillna(0)

        # Get final feature names
        feature_names = list(X.columns)

        return X.values, feature_names

    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if not self.model:
            return {}

        if self.use_xgboost and hasattr(self.model, "get_score"):
            # XGBoost feature importance
            importance_scores = self.model.get_score(importance_type="gain")
            return {k: float(v) for k, v in importance_scores.items()}
        elif hasattr(self.model, "feature_importances_"):
            # sklearn feature importance
            importance_scores = self.model.feature_importances_
            importance_dict = {}
            for i, feature_name in enumerate(self.feature_names):
                if i < len(importance_scores):
                    importance_dict[feature_name] = float(importance_scores[i])
            return importance_dict
        else:
            return {}

    def train(
        self, training_data: pd.DataFrame, hyperparams: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Train the XGBoost model on historical market data.

        Args:
            training_data: DataFrame with market features and outcomes
            hyperparams: Optional XGBoost hyperparameters
        """
        logger.info(f"Training XGBoost model on {len(training_data)} samples...")

        # Default hyperparameters
        default_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "early_stopping_rounds": 10,
        }

        if hyperparams:
            default_params.update(hyperparams)

        # Prepare features and target
        X_train, y_train, feature_names = self._prepare_features_for_training(
            training_data
        )

        # Store feature names for prediction
        self.feature_names = feature_names

        if self.use_xgboost:
            # Use XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)

            # Train model
            self.model = xgb.train(
                default_params,
                dtrain,
                num_boost_round=default_params["n_estimators"],
                early_stopping_rounds=default_params.get("early_stopping_rounds"),
                verbose_eval=False,
            )

            # Save the model
            self.model.save_model(str(self.model_path))
            logger.info(f"✅ XGBoost model trained and saved to {self.model_path}")
        else:
            # Use sklearn GradientBoostingClassifier
            from sklearn.ensemble import GradientBoostingClassifier

            self.model = GradientBoostingClassifier(
                n_estimators=default_params.get("n_estimators", 100),
                learning_rate=default_params.get("learning_rate", 0.1),
                max_depth=default_params.get("max_depth", 6),
                subsample=default_params.get("subsample", 0.8),
                random_state=default_params.get("random_state", 42),
            )

            self.model.fit(X_train, y_train)

            # Save using joblib
            import joblib

            joblib.dump(self.model, self.model_path)
            logger.info(
                f"✅ Sklearn GradientBoosting model trained and saved to {self.model_path}"
            )

        # Mark as trained
        self.trained = True

        # Log feature importance
        importance = self._get_feature_importance()
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info(f"Top 10 features: {top_features}")

    def predict(self, market_data: Dict[str, Any]) -> StrategyResult:
        """
        Make a prediction for a specific market using the trained model.

        Args:
            market_data: Dictionary containing market information

        Returns:
            StrategyResult with prediction details
        """
        if not self.trained or not self.model:
            raise ValueError("Model must be trained before making predictions")

        # Prepare features for prediction using custom method
        X_pred, feature_names = self._prepare_features_for_prediction(market_data)

        if self.use_xgboost:
            # XGBoost prediction
            dpred = xgb.DMatrix(X_pred, feature_names=feature_names)
            predicted_prob = float(self.model.predict(dpred)[0])
        else:
            # sklearn prediction
            predicted_prob = float(
                self.model.predict_proba(X_pred)[0][1]
            )  # Probability of positive class

        # Get current market-implied probability (try different field names)
        market_prob = 0.5  # default
        if "outcome_prices" in market_data and isinstance(
            market_data["outcome_prices"], list
        ):
            market_prob = float(market_data["outcome_prices"][0])
        elif "yes_price" in market_data:
            market_prob = float(market_data["yes_price"])
        elif "market_prob" in market_data:
            market_prob = float(market_data["market_prob"])

        # Calculate edge (expected value per dollar bet)
        edge = self.calculate_edge(predicted_prob, market_prob)

        # Determine recommendation
        confidence = min(abs(predicted_prob - market_prob) * 2, 1.0)  # Scale to 0-1

        if edge > 0.02:  # >2% edge
            if predicted_prob > market_prob:
                recommendation = "YES"
            else:
                recommendation = "NO"
        else:
            recommendation = "PASS"

        # Calculate position size
        position_size = self.kelly_criterion(edge, confidence)

        # Create result
        result = StrategyResult(
            market_id=market_data.get("market_id", market_data.get("id", "unknown")),
            market_question=market_data.get("question", "Unknown market"),
            predicted_probability=predicted_prob,
            confidence=confidence,
            edge=edge,
            recommended_bet=recommendation,
            position_size=position_size,
            expected_value=edge,  # Simplified
            reasoning=f"XGBoost predicts {predicted_prob:.3f} probability, market shows {market_prob:.3f}. Edge: {edge:.1%}",
            features_used=feature_names,
            model_name=self.name,
            timestamp=datetime.now(),
        )

        return result

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance scores from the trained model."""
        return self._get_feature_importance()

    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        if not self.model:
            raise ValueError("No trained model to save")

        if self.use_xgboost:
            self.model.save_model(filepath)
        else:
            import joblib

            joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        if self.use_xgboost:
            self.model = xgb.Booster()
            self.model.load_model(filepath)
        else:
            import joblib

            self.model = joblib.load(filepath)
        self.trained = True
        logger.info(f"Model loaded from {filepath}")

    def create_training_dataset(
        self, days_back: int = 365, min_volume: float = 1000
    ) -> pd.DataFrame:
        """
        Create a training dataset using the data ingestion pipeline.

        Args:
            days_back: Days of historical data to use
            min_volume: Minimum volume threshold

        Returns:
            DataFrame ready for training
        """
        logger.info("Creating training dataset for XGBoost...")

        # Use the existing data ingestion pipeline
        training_data = self.data_ingestor.create_training_dataset(
            days_back=days_back,
            min_volume=min_volume,
            include_unresolved=False,  # Only resolved markets for supervised learning
        )

        logger.info(f"Created training dataset with {len(training_data)} samples")
        return training_data

    def evaluate_model(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.

        Args:
            test_data: Test dataset

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.trained or not self.model:
            raise ValueError("Model must be trained before evaluation")

        logger.info(f"Evaluating model on {len(test_data)} test samples...")

        # Prepare test features
        X_test, y_test, _ = self._prepare_features_for_training(test_data)

        # Get predictions
        if self.use_xgboost:
            dtest = xgb.DMatrix(X_test, label=y_test)
            y_pred_prob = self.model.predict(dtest)
        else:
            # sklearn prediction
            y_pred_prob = self.model.predict_proba(X_test)[
                :, 1
            ]  # Probability of positive class

        y_pred = (y_pred_prob > 0.5).astype(int)

        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            brier_score_loss,
            log_loss,
        )

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_prob),
            "brier_score": brier_score_loss(y_test, y_pred_prob),
            "log_loss": log_loss(y_test, y_pred_prob),
            "sample_size": len(test_data),
        }

        # Market-specific metrics
        test_data_copy = test_data.copy()
        test_data_copy["predicted_prob"] = y_pred_prob
        # Get market probability (try different column names)
        if "yes_price" in test_data_copy.columns:
            test_data_copy["market_prob"] = test_data_copy["yes_price"]
        elif "market_prob" in test_data_copy.columns:
            pass  # already exists
        else:
            test_data_copy["market_prob"] = 0.5  # default

        # Calculate edge distribution
        test_data_copy["edge"] = test_data_copy.apply(
            lambda row: self.calculate_edge(row["predicted_prob"], row["market_prob"]),
            axis=1,
        )

        edge_metrics = {
            "mean_edge": test_data_copy["edge"].mean(),
            "median_edge": test_data_copy["edge"].median(),
            "positive_edge_pct": (test_data_copy["edge"] > 0).mean(),
            "large_edge_pct": (test_data_copy["edge"] > 0.05).mean(),  # >5% edge
        }

        metrics.update(edge_metrics)

        logger.info(
            f"Evaluation complete. Accuracy: {metrics['accuracy']:.3f}, ROC-AUC: {metrics['roc_auc']:.3f}"
        )
        logger.info(
            f"Mean edge: {metrics['mean_edge']:.1%}, Positive edge %: {metrics['positive_edge_pct']:.1%}"
        )

        return metrics

    def backtest_strategy(
        self, historical_data: pd.DataFrame, initial_capital: float = 10000.0
    ) -> Dict[str, Any]:
        """
        Backtest the strategy on historical data.

        Args:
            historical_data: Historical market data
            initial_capital: Starting capital for simulation

        Returns:
            Backtest results including performance metrics
        """
        logger.info(
            f"Backtesting strategy on {len(historical_data)} historical markets..."
        )

        capital = initial_capital
        trades = []
        portfolio_value = [initial_capital]

        for _, market in historical_data.iterrows():
            try:
                # Convert market row to dict for prediction
                market_dict = {str(k): v for k, v in market.to_dict().items()}

                # Make prediction
                result = self.predict(market_dict)

                # Simulate trade if recommendation is made
                if result.recommended_bet != "PASS" and result.position_size > 0:
                    # Calculate stake
                    stake = min(
                        capital * result.position_size, capital * 0.1
                    )  # Max 10% per trade

                    # Get market probability for payout calculation
                    market_prob = 0.5
                    if "yes_price" in market_dict:
                        market_prob = market_dict["yes_price"]
                    elif "market_prob" in market_dict:
                        market_prob = market_dict["market_prob"]
                    elif "outcome_prices" in market_dict and isinstance(
                        market_dict["outcome_prices"], list
                    ):
                        market_prob = float(market_dict["outcome_prices"][0])

                    if result.recommended_bet == "YES":
                        # Bet on YES outcome
                        payout_ratio = 1.0 / market_prob if market_prob > 0 else 2.0
                    else:
                        # Bet on NO outcome - assume symmetric market
                        payout_ratio = (
                            1.0 / (1 - market_prob) if (1 - market_prob) > 0 else 2.0
                        )

                    # Determine if trade was profitable
                    actual_outcome = market_dict.get(
                        "actual_outcome", market_dict.get("target", 0)
                    )
                    expected_outcome = 1 if result.predicted_probability > 0.5 else 0

                    if actual_outcome == expected_outcome:
                        # Winning trade
                        profit = stake * (payout_ratio - 1)
                        capital += profit
                    else:
                        # Losing trade
                        capital -= stake

                    # Record trade
                    trade = {
                        "market_id": result.market_id,
                        "prediction": result.predicted_probability,
                        "market_prob": market_prob,
                        "recommendation": result.recommended_bet,
                        "stake": stake,
                        "outcome": actual_outcome,
                        "profit": (
                            profit if actual_outcome == expected_outcome else -stake
                        ),
                        "capital_after": capital,
                    }
                    trades.append(trade)

                portfolio_value.append(capital)

            except Exception as e:
                logger.warning(
                    f"Error processing market {market.get('market_id', 'unknown')}: {e}"
                )
                continue

        # Calculate performance metrics
        final_capital = capital
        total_return = (final_capital - initial_capital) / initial_capital
        num_trades = len(trades)
        win_rate = sum(1 for t in trades if t["profit"] > 0) / max(num_trades, 1)

        # Sharpe ratio (simplified)
        returns = np.diff(portfolio_value) / portfolio_value[:-1]
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        results = {
            "initial_capital": initial_capital,
            "final_capital": final_capital,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "num_trades": num_trades,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": self._calculate_max_drawdown(portfolio_value),
            "trades": trades[:100],  # Limit trade history for storage
        }

        logger.info(
            f"Backtest complete: {total_return:.1%} return, {num_trades} trades, {win_rate:.1%} win rate"
        )
        return results

    def run_full_pipeline_from_data(
        self, training_data: pd.DataFrame, test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Run the complete ML pipeline using pre-loaded training data.

        Args:
            training_data: Pre-loaded training dataset
            test_size: Fraction for testing

        Returns:
            Dictionary with pipeline results
        """
        logger.info("🚀 Starting XGBoost ML pipeline with provided data...")

        if len(training_data) == 0:
            raise ValueError("No training data provided")

        # 2. Split train/test
        train_data = training_data.sample(frac=1 - test_size, random_state=42)
        test_data = training_data.drop(train_data.index)

        logger.info(
            f"Step 2: Split into {len(train_data)} train, {len(test_data)} test samples"
        )

        # 3. Train model
        logger.info("Step 3: Training XGBoost model...")
        self.train(train_data)

        # 4. Evaluate model
        logger.info("Step 4: Evaluating model...")
        eval_metrics = self.evaluate_model(test_data)

        # 5. Backtest strategy (if possible)
        logger.info("Step 5: Backtesting strategy...")
        try:
            backtest_results = self.backtest_strategy(test_data)
        except Exception as e:
            logger.warning(f"Backtesting failed: {e}")
            backtest_results = {"error": str(e)}

        # 6. Get feature importance
        feature_importance = self.get_feature_importance()

        results = {
            "training_samples": len(train_data),
            "test_samples": len(test_data),
            "evaluation_metrics": eval_metrics,
            "backtest_results": backtest_results,
            "feature_importance": feature_importance,
            "model_path": str(self.model_path),
            "timestamp": datetime.now().isoformat(),
        }

        logger.info("✅ XGBoost pipeline completed successfully")
        logger.info(f"Model ROC-AUC: {eval_metrics.get('roc_auc', 0):.3f}")

        return results

    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown from portfolio values."""
        if len(portfolio_values) < 2:
            return 0.0

        peak = portfolio_values[0]
        max_drawdown = 0.0

        for value in portfolio_values[1:]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def run_full_pipeline(
        self, days_back: int = 365, min_volume: float = 1000, test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Run the complete ML pipeline: data ingestion, training, evaluation, and backtesting.

        Args:
            days_back: Days of historical data to use
            min_volume: Minimum volume threshold
            test_size: Fraction of data to use for testing

        Returns:
            Dictionary with pipeline results
        """
        logger.info("🚀 Starting XGBoost ML pipeline...")

        # 1. Create training dataset
        logger.info("Step 1: Creating training dataset...")
        training_data = self.create_training_dataset(
            days_back=days_back, min_volume=min_volume
        )

        if len(training_data) == 0:
            raise ValueError("No training data available")

        # 2. Split train/test
        train_data = training_data.sample(frac=1 - test_size, random_state=42)
        test_data = training_data.drop(train_data.index)

        logger.info(
            f"Step 2: Split into {len(train_data)} train, {len(test_data)} test samples"
        )

        # 3. Train model
        logger.info("Step 3: Training XGBoost model...")
        self.train(train_data)

        # 4. Evaluate model
        logger.info("Step 4: Evaluating model...")
        eval_metrics = self.evaluate_model(test_data)

        # 5. Backtest strategy
        logger.info("Step 5: Backtesting strategy...")
        backtest_results = self.backtest_strategy(test_data)

        # 6. Get feature importance
        feature_importance = self.get_feature_importance()

        results = {
            "training_samples": len(train_data),
            "test_samples": len(test_data),
            "evaluation_metrics": eval_metrics,
            "backtest_results": backtest_results,
            "feature_importance": feature_importance,
            "model_path": str(self.model_path),
            "timestamp": datetime.now().isoformat(),
        }

        logger.info("✅ XGBoost pipeline completed successfully")
        logger.info(
            f"Model performance: ROC-AUC = {eval_metrics.get('roc_auc', 0):.3f}"
        )
        logger.info(
            f"Backtest return: {backtest_results.get('total_return_pct', 0):.1f}%"
        )

        return results
