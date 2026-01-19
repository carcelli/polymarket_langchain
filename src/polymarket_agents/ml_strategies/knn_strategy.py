import numpy as np
from typing import Dict, Any, List
from collections import Counter
from .registry import register_strategy
from .base_strategy import MLBettingStrategy, StrategyResult
from .evaluation import classification_report

@register_strategy("knn_edge_detector")
class KNNEdgeDetector(MLBettingStrategy):
    """k-NN strategy using Euclidean distance with distance-weighted voting."""
    
    def __init__(self, k: int = 5, analyzer=None):
        super().__init__("KNNEdgeDetector", analyzer)
        self.k = k
        self.historical_data = None  # Set during training

    def train(self, training_data: np.ndarray, resolutions: np.ndarray) -> None:
        """Store historical markets and resolutions (0=NO, 1=YES)."""
        # training_data expected to be (N, features)
        # resolutions expected to be (N,)
        if len(training_data) != len(resolutions):
            raise ValueError("Training data and resolutions must have same length")
            
        self.historical_data = np.column_stack((training_data, resolutions))
        self.is_trained = True

    def predict(self, market_data: Dict[str, Any]) -> StrategyResult:
        if not hasattr(self, 'is_trained') or not self.is_trained or self.historical_data is None:
            # Fallback if not trained
            return StrategyResult(
                market_id=market_data.get('id', 'unknown'),
                market_question=market_data.get('question', ''),
                predicted_probability=0.5,
                confidence=0.0,
                edge=0.0,
                recommended_bet="PASS",
                position_size=0.0,
                expected_value=0.0,
                reasoning="Model not trained",
                features_used=[],
                model_name=self.name,
                timestamp=np.datetime64('now')
            )

        features = self.prepare_features(market_data).flatten()
        
        # Calculate distances using numpy
        # historical_data columns: [features..., resolution]
        # We compute distance against all rows' features
        X_train = self.historical_data[:, :-1]
        y_train = self.historical_data[:, -1]
        
        # Vectorized distance calculation
        # shape (N,)
        dists = np.linalg.norm(X_train - features, axis=1)
        
        # Get indices of k nearest neighbors
        # argsort returns indices that would sort the array
        nearest_indices = np.argsort(dists)[:self.k]
        
        neighbors = []
        for idx in nearest_indices:
            neighbors.append((dists[idx], y_train[idx]))
        
        # Distance-weighted voting
        weights = Counter()
        for dist, label in neighbors:
            weights[label] += 1 / (dist + 1e-8)  # avoid division by zero
        
        total_weight = sum(weights.values())
        if total_weight == 0:
            pred_prob = 0.5
        else:
            pred_prob = weights[1] / total_weight
            
        current_prob = float(market_data.get('outcome_prices', ['0.5', '0.5'])[0])
        # Sometimes current_prob might be a string in the dict, ensure float. 
        # But prepare_features handles some of this. Let's trust input or sanitize.
        
        edge = abs(pred_prob - current_prob)
        
        # Confidence logic from prompt
        # max(weighted_vote_ratio, 0.5)
        if total_weight > 0:
            confidence = max(weights.most_common(1)[0][1] / total_weight if weights else 0.5, 0.5)
        else:
            confidence = 0.5
        
        recommendation = "BUY_YES" if pred_prob > 0.5 + edge/2 else "BUY_NO" if pred_prob < 0.5 - edge/2 else "HOLD"
        
        return StrategyResult(
            market_id=market_data.get('id', 'unknown'),
            market_question=market_data.get('question', ''),
            predicted_probability=pred_prob,
            confidence=confidence,
            edge=edge,
            recommended_bet=recommendation,
            position_size=self.kelly_criterion(edge, confidence),
            expected_value=edge * confidence,
            reasoning=f"k-NN ({self.k} neighbors) weighted vote: {pred_prob:.1%} YES probability",
            features_used=self.feature_columns,
            model_name=self.name,
            timestamp=np.datetime64('now')
        )

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance scores. Not applicable for basic k-NN."""
        return {col: 1.0/len(self.feature_columns) for col in self.feature_columns} if self.feature_columns else {}
