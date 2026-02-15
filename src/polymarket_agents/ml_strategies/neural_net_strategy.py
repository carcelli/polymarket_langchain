from __future__ import annotations

import numpy as np
from scipy.special import expit as sigmoid
from scipy.stats import truncnorm
from typing import List, Dict, Any
from .registry import register_strategy
from .base_strategy import MLBettingStrategy, StrategyResult


def truncated_normal(
    mean: float = 0.0, sd: float = 1.0, low: float = -0.5, upp: float = 0.5
):
    """Xavier-like initialization scaled by layer size."""
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class NeuralNetwork:
    """Multi-layer perceptron with bias, configurable structure."""

    def __init__(
        self,
        structure: List[int],  # e.g., [20, 10, 10, 1]
        learning_rate: float = 0.1,
        bias: float | None = 1.0,  # None disables bias
    ):
        self.structure = structure
        self.lr = learning_rate
        self.bias_val = bias
        self.use_bias = bias is not None
        self._build_weights()

    def _build_weights(self):
        self.weights: List[np.ndarray] = []
        bias_nodes = 1 if self.use_bias else 0
        for i in range(1, len(self.structure)):
            nodes_in = self.structure[i - 1] + bias_nodes
            nodes_out = self.structure[i]
            # Use Xavier initialization range
            rad = 1 / np.sqrt(nodes_in)
            X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
            wm = X.rvs((nodes_out, nodes_in))
            self.weights.append(wm)

    def forward(self, inputs: np.ndarray) -> List[np.ndarray]:
        """Forward pass returning all layer activations."""
        # Ensure inputs is a column vector
        act = np.array(inputs).reshape(-1, 1)
        activations = [act]
        for i, w in enumerate(self.weights):
            if self.use_bias:
                # Add bias row to activation
                act = np.vstack((act, [[self.bias_val]]))

            # Weighted sum
            net = np.dot(w, act)
            # Sigmoid activation
            act = sigmoid(net)
            activations.append(act)
        return activations

    def train_single(self, inputs: np.ndarray, target: np.ndarray):
        """
        Single example backprop update.
        target should be shape (n_outputs,)
        """
        acts = self.forward(inputs)
        target = np.array(target).reshape(-1, 1)

        # Output layer error
        # derivative of sigmoid is output * (1 - output)
        output_act = acts[-1]
        errors = target - output_act

        # Backpropagate
        for i in reversed(range(len(self.weights))):
            output = acts[i + 1]
            # Gradient: error * derivative
            delta = errors * output * (1 - output)

            # Input to this layer (activation from previous)
            input_act = acts[i]
            if self.use_bias:
                input_act = np.vstack((input_act, [[self.bias_val]]))

            # Weight update: learning_rate * delta * input_transposed
            dw = self.lr * np.dot(delta, input_act.T)
            self.weights[i] += dw

            # Calculate error for previous layer (if not input layer)
            if i > 0:
                # Propagate error back
                # weights[i] shape is (nodes_out, nodes_in)
                # delta shape is (nodes_out, 1)
                # We need errors for nodes_in.
                # error_prev = W.T dot delta
                errors = np.dot(self.weights[i].T, delta)

                # If using bias, the bias node in the previous layer doesn't have an error signal propagated *to* it
                # (it doesn't change based on error), or rather we don't need to calculate it for further backprop
                # because the layer before that doesn't connect to this bias node.
                # But wait, input_act included bias.
                # errors calculated here correspond to (nodes_in + bias).
                # We need to remove the error term for the bias node before going to next layer back
                if self.use_bias:
                    errors = errors[:-1]  # Remove bias error contribution

    def predict(self, inputs: np.ndarray) -> float:
        """Return YES probability (assuming single output node)."""
        return self.forward(inputs)[-1].flatten()[0]


@register_strategy("neural_net_predictor")
class NeuralNetStrategy(MLBettingStrategy):
    """NN strategy for market YES probability."""

    def __init__(
        self,
        structure: List[int] = [30, 20, 10, 1],
        lr: float = 0.1,
        epochs: int = 20,
        bias: float | None = 1.0,
    ):
        super().__init__("NeuralNetPredictor")
        self.structure = structure
        self.lr = lr
        self.epochs = epochs
        self.bias = bias
        self.nn = NeuralNetwork(structure, lr, bias)
        self.trained = False

    def train(self, features: np.ndarray, resolutions: np.ndarray):
        """Train over epochs."""
        # features: (N_samples, N_features)
        # resolutions: (N_samples,)

        # Adjust input layer size if necessary to match feature count
        input_dim = features.shape[1]
        if self.structure[0] != input_dim:
            print(f"Adjusting NN input layer from {self.structure[0]} to {input_dim}")
            self.structure[0] = input_dim
            self.nn = NeuralNetwork(self.structure, self.lr, self.bias)

        print(f"Training Neural Network ({self.epochs} epochs)...")
        for epoch in range(self.epochs):
            loss = 0
            for x, y in zip(features, resolutions):
                self.nn.train_single(x, np.array([y]))
                # specific simple loss tracking
                pred = self.nn.predict(x)
                loss += (y - pred) ** 2

            if epoch % 5 == 0:
                print(f"Epoch {epoch}: MSE = {loss / len(features):.4f}")

        self.trained = True

    def predict(self, market_data: Dict[str, Any]) -> StrategyResult:
        if not self.trained:
            # Fallback
            return StrategyResult(
                market_id=market_data.get("id", "unknown"),
                market_question=market_data.get("question", ""),
                predicted_probability=0.5,
                confidence=0.0,
                edge=0.0,
                recommended_bet="PASS",
                position_size=0.0,
                expected_value=0.0,
                reasoning="Model not trained",
                features_used=[],
                model_name=self.name,
                timestamp=np.datetime64("now"),
            )

        # Prepare features
        # Note: prepare_features returns (1, n_features) array
        feats = self.prepare_features(market_data).flatten()

        # Predict
        pred_prob = self.nn.predict(feats)

        current_prob = float(market_data.get("outcome_prices", ["0.5", "0.5"])[0])

        edge = abs(pred_prob - current_prob)
        # Confidence scales with how extreme the prediction is (closer to 0 or 1)
        confidence = abs(pred_prob - 0.5) * 2

        recommendation = "HOLD"
        if pred_prob > current_prob + 0.02:  # 2% threshold
            recommendation = "BUY_YES"
        elif pred_prob < current_prob - 0.02:
            recommendation = "BUY_NO"

        return StrategyResult(
            market_id=market_data.get("id", "unknown"),
            market_question=market_data.get("question", ""),
            predicted_probability=float(pred_prob),
            confidence=float(confidence),
            edge=float(edge),
            recommended_bet=recommendation,
            position_size=self.kelly_criterion(edge, confidence),
            expected_value=edge * confidence,
            reasoning=f"NN ({self.epochs} epochs) predicts {pred_prob:.1%} YES",
            features_used=self.feature_columns,
            model_name=self.name,
            timestamp=np.datetime64("now"),
        )

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Return approximate feature importance based on input weights.
        Sum of absolute weights connecting input to first hidden layer.
        """
        if not self.trained:
            return {}

        input_weights = self.nn.weights[0]  # Shape (hidden_nodes, input_nodes + bias)

        # Exclude bias weight if present
        if self.nn.use_bias:
            # input_weights corresponds to [node_0_weights, ..., node_n_weights]
            # actually weights shape is (nodes_out, nodes_in).
            # nodes_in includes bias at the end if we stacked it that way?
            # Let's check forward: act = np.vstack((act, [[self.bias_val]]))
            # So bias is the LAST element of the input vector.
            weight_magnitudes = np.sum(np.abs(input_weights[:, :-1]), axis=0)
        else:
            weight_magnitudes = np.sum(np.abs(input_weights), axis=0)

        # Normalize
        total = np.sum(weight_magnitudes)
        if total == 0:
            return {}

        importance = weight_magnitudes / total

        return {
            col: float(imp)
            for col, imp in zip(self.feature_columns, importance)
            if col  # Ensure column name exists
        }
