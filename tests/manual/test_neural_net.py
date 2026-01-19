import numpy as np
import sys
import os

# Add src and scripts/workflows to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts/workflows')))

from polymarket_agents.ml_strategies.neural_net_strategy import NeuralNetStrategy

def test_neural_net_xor():
    print("Testing Neural Net on XOR problem...")
    
    # XOR problem (non-linear)
    # 0,0 -> 0
    # 0,1 -> 1
    # 1,0 -> 1
    # 1,1 -> 0
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([0, 1, 1, 0])
    
    # Structure: 2 input, 4 hidden, 1 output
    # High learning rate for quick convergence on simple problem
    strategy = NeuralNetStrategy(structure=[2, 5, 1], lr=0.5, epochs=1000)
    
    # Monkey patch prepare_features to just return the input
    strategy.prepare_features = lambda market_data: np.array(market_data['features']).reshape(1, -1)
    
    strategy.train(X, y)
    
    print("\nPredictions:")
    correct = 0
    for i in range(len(X)):
        market_data = {
            'id': f'test_{i}', 
            'features': X[i], 
            'outcome_prices': ['0.5', '0.5']
        }
        res = strategy.predict(market_data)
        pred = res.predicted_probability
        rounded = round(pred)
        print(f"Input: {X[i]}, Target: {y[i]}, Pred: {pred:.4f} ({rounded})")
        if rounded == y[i]:
            correct += 1
            
    accuracy = correct / len(X)
    print(f"Accuracy: {accuracy:.2f}")
    
    # XOR usually requires more tuning or epochs, but should get at least 75% or 100% 
    # if initialized well.
    assert accuracy >= 0.75, "Failed to learn XOR (or close enough)"
    
    print("Neural Net XOR test passed!")

def test_feature_importance():
    print("\nTesting Feature Importance...")
    strategy = NeuralNetStrategy(structure=[2, 2, 1], epochs=1)
    
    X = np.array([[0,0], [0,1]])
    y = np.array([0, 1])
    
    # Mock feature columns
    strategy.feature_columns = ['feat_a', 'feat_b']
    
    # Monkey patch prepare_features
    strategy.prepare_features = lambda market_data: np.array(market_data['features']).reshape(1, -1)
    
    strategy.train(X, y)
    
    imp = strategy.get_feature_importance()
    print("Feature Importance:", imp)
    
    assert 'feat_a' in imp
    assert 'feat_b' in imp
    assert abs(sum(imp.values()) - 1.0) < 1e-6
    
    print("Feature Importance test passed!")

if __name__ == "__main__":
    test_neural_net_xor()
    test_feature_importance()
