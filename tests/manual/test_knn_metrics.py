import numpy as np
import sys
import os

# Add src and scripts/workflows to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts/workflows')))

from polymarket_agents.ml_strategies.evaluation import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
)
from polymarket_agents.ml_strategies.knn_strategy import KNNEdgeDetector

def test_metrics():
    print("Testing metrics...")
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 0, 1])
    
    # TP: 3 (indices 0, 3, 5)
    # TN: 2 (indices 1, 4)
    # FP: 0
    # FN: 1 (index 2)
    
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    assert cm[0, 0] == 2 # TN
    assert cm[1, 1] == 3 # TP
    assert cm[0, 1] == 0 # FP
    assert cm[1, 0] == 1 # FN
    
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.2f}")
    assert acc == 5/6
    
    prec = precision_score(y_true, y_pred)
    print(f"Precision: {prec:.2f}")
    assert prec == 3/3 # 1.0
    
    rec = recall_score(y_true, y_pred)
    print(f"Recall: {rec:.2f}")
    assert rec == 3/4 # 0.75
    
    f1 = f1_score(y_true, y_pred)
    print(f"F1: {f1:.2f}")
    expected_f1 = 2 * (1.0 * 0.75) / (1.0 + 0.75)
    assert abs(f1 - expected_f1) < 1e-6

    print("Metrics tests passed!")

def test_knn():
    print("\nTesting KNN Strategy...")
    strategy = KNNEdgeDetector(k=3)
    
    # Synthetic data: 2 features
    # Cluster 0: around (0,0) -> NO (0)
    # Cluster 1: around (5,5) -> YES (1)
    
    X_train = np.array([
        [0.1, 0.1], [0.2, 0.2], [0, 0.1], # Cluster 0
        [5.1, 5.1], [5.2, 5.0], [5.0, 5.2]  # Cluster 1
    ])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    
    # Monkey patch prepare_features for this instance
    def mock_prepare_features(market_data):
        return np.array(market_data['features']).reshape(1, -1)
    
    strategy.prepare_features = mock_prepare_features
    
    strategy.train(X_train, y_train)
    
    # Test point near Cluster 1
    test_market = {
        'id': 'test1',
        'question': 'Will it happen?',
        'features': [5.1, 5.15],
        'outcome_prices': ['0.4', '0.6'] # Market thinks 40% YES, we should think ~100% YES
    }
    
    result = strategy.predict(test_market)
    print("Prediction Result:", result)
    
    assert result.predicted_probability > 0.8
    assert result.recommended_bet == "BUY_YES"
    
    # Test point near Cluster 0
    test_market_0 = {
        'id': 'test0',
        'question': 'Will it happen?',
        'features': [0.15, 0.15],
        'outcome_prices': ['0.6', '0.4'] # Market thinks 60% YES, we should think ~0% YES
    }
    
    result_0 = strategy.predict(test_market_0)
    print("Prediction Result 0:", result_0)
    
    assert result_0.predicted_probability < 0.2
    assert result_0.recommended_bet == "BUY_NO"

    print("KNN Strategy tests passed!")

if __name__ == "__main__":
    test_metrics()
    test_knn()
