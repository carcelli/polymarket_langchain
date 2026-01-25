"""
Neural Network Implementation

Multi-layer perceptron with backpropagation for classification tasks.
Designed for integration with Polymarket agents for market prediction.
"""

import numpy as np
from typing import Optional, Tuple, Union
from .utils import (
    truncated_normal,
    sigmoid,
    softmax,
    cross_entropy_loss,
    confusion_matrix,
    precision_recall,
)


class NeuralNetwork:
    """
    Multi-layer perceptron with backpropagation.

    Architecture: Input → Hidden → Output
    Supports sigmoid and softmax activations.
    """

    def __init__(
        self,
        n_inputs: int,
        n_hidden: int,
        n_outputs: int,
        lr: float = 0.1,
        bias: float = 1.0,
        use_softmax: bool = True,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize neural network.

        Args:
            n_inputs: Number of input features
            n_hidden: Number of hidden neurons
            n_outputs: Number of output classes
            lr: Learning rate
            bias: Bias term value
            use_softmax: Whether to use softmax (True) or sigmoid (False) output
            random_seed: Random seed for reproducible weight initialization
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        self.lr = lr
        self.bias = bias
        self.use_softmax = use_softmax
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        # Xavier/Glorot initialization for better gradient flow
        # Scale by sqrt(n_inputs) to prevent vanishing/exploding gradients
        input_rad = 1 / np.sqrt(n_inputs)
        hidden_rad = 1 / np.sqrt(n_hidden)

        # Input → Hidden weights (including bias column)
        X = truncated_normal(low=-input_rad, upp=input_rad)
        self.wih = X.rvs((n_hidden, n_inputs + 1))  # +1 for bias

        # Hidden → Output weights (including bias row)
        X = truncated_normal(low=-hidden_rad, upp=hidden_rad)
        self.who = X.rvs((n_outputs, n_hidden + 1))  # +1 for bias

    @classmethod
    def from_layers(
        cls,
        layer_sizes: list[int],
        lr: float = 0.1,
        bias: float = 1.0,
        use_softmax: bool = True,
        random_seed: Optional[int] = None,
    ) -> "NeuralNetwork":
        """
        Create a multi-layer neural network with arbitrary architecture.

        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
                         e.g., [784, 200, 100, 10] for MNIST
            lr: Learning rate
            bias: Bias term value
            use_softmax: Whether to use softmax (True) or sigmoid (False) output
            random_seed: Random seed for reproducible weight initialization

        Returns:
            NeuralNetwork instance with multi-layer architecture

        Example:
            nn = NeuralNetwork.from_layers([784, 200, 100, 10])  # MNIST
        """
        if len(layer_sizes) < 2:
            raise ValueError("Must specify at least input and output layers")

        if random_seed is not None:
            np.random.seed(random_seed)

        # Create instance with dummy single-layer params
        instance = cls.__new__(cls)

        instance.layer_sizes = layer_sizes
        instance.lr = lr
        instance.bias = bias
        instance.use_softmax = use_softmax

        # Xavier/Glorot initialization for each layer transition
        instance.weights = []
        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            rad = 1 / np.sqrt(n_in)  # Xavier initialization

            X = truncated_normal(low=-rad, upp=rad)
            # +1 for bias column/row
            w = X.rvs((n_out, n_in + 1))
            instance.weights.append(w)

        # Backward compatibility attributes
        instance.n_inputs = layer_sizes[0]
        instance.n_hidden = layer_sizes[1] if len(layer_sizes) > 2 else layer_sizes[1]
        instance.n_outputs = layer_sizes[-1]

        # Single-layer backward compatibility (used by existing methods)
        if len(layer_sizes) == 3:  # [input, hidden, output]
            instance.wih = instance.weights[0]
            instance.who = instance.weights[1]

        return instance

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Forward pass through the network.

        Args:
            x: Input features [n_inputs]

        Returns:
            Tuple of (output, activations) where:
            - output: Network output [n_outputs]
            - activations: List of activations for each layer (for backprop)
        """
        # Handle both single-layer and multi-layer architectures
        if hasattr(self, "weights") and len(self.weights) > 2:
            # Multi-layer forward pass
            output, activations = self._forward_multilayer(x)
            return output, activations
        else:
            # Single-layer forward pass (backward compatibility)
            output = self._forward_singlelayer(x)
            return output

    def _forward_singlelayer(self, x: np.ndarray) -> np.ndarray:
        """Single-layer forward pass for backward compatibility."""
        # Add bias to input
        x_with_bias = np.concatenate([x, [self.bias]])[:, None]  # Column vector

        # Hidden layer: weighted sum → sigmoid
        hidden_input = self.wih @ x_with_bias
        hidden_output = sigmoid(hidden_input.ravel())

        # Add bias to hidden layer
        hidden_with_bias = np.concatenate([hidden_output, [self.bias]])[:, None]

        # Output layer: weighted sum → activation
        output_input = self.who @ hidden_with_bias

        if self.use_softmax:
            return softmax(output_input.ravel())
        else:
            return sigmoid(output_input.ravel())

    def _forward_multilayer(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Multi-layer forward pass."""
        activations = []
        act = np.concatenate([x, [self.bias]])  # Include bias
        activations.append(act)

        # Forward through hidden layers
        for w in self.weights[:-1]:
            act = w @ act  # Weighted sum
            act = sigmoid(act)  # Activation
            act = np.concatenate([act, [self.bias]])  # Add bias for next layer
            activations.append(act)

        # Output layer
        output_input = self.weights[-1] @ act
        if self.use_softmax:
            output = softmax(output_input)
        else:
            output = sigmoid(output_input)

        # Store output activations (without bias) for backprop
        # Note: Output layer doesn't get bias added since it's the final layer
        output_activations = output  # Store activated output
        activations.append(output_activations)

        return output, activations

    def backward(self, x: np.ndarray, target: np.ndarray) -> float:
        """
        Backward pass - compute gradients and update weights.

        Args:
            x: Input features [n_inputs]
            target: Target output [n_outputs]

        Returns:
            Loss value
        """
        # Forward pass to get activations
        x_with_bias = np.concatenate([x, [self.bias]])[:, None]
        hidden_input = self.wih @ x_with_bias
        hidden_output = sigmoid(hidden_input.ravel())
        hidden_with_bias = np.concatenate([hidden_output, [self.bias]])[:, None]

        output_input = self.who @ hidden_with_bias
        if self.use_softmax:
            output_activation = softmax(output_input.ravel())
        else:
            output_activation = sigmoid(output_input.ravel())

        # Calculate loss
        if self.use_softmax:
            loss = cross_entropy_loss(
                output_activation.reshape(1, -1), target.reshape(1, -1)
            )
        else:
            # Binary cross-entropy for sigmoid
            loss = -np.sum(
                target * np.log(output_activation + 1e-15)
                + (1 - target) * np.log(1 - output_activation + 1e-15)
            )

        # Output layer error (delta)
        if self.use_softmax:
            # Softmax derivative: Jacobian matrix
            s = output_activation
            jacobian = np.diag(s) - np.outer(s, s)
            output_error = jacobian @ (target - output_activation)
        else:
            # Sigmoid derivative: output * (1 - output)
            output_error = (
                (target - output_activation)
                * output_activation
                * (1 - output_activation)
            )

        # Hidden layer error (backpropagate)
        # Remove bias term from weights for backprop
        who_no_bias = self.who[:, :-1]  # Remove bias column
        hidden_error = (
            (who_no_bias.T @ output_error) * hidden_output * (1 - hidden_output)
        )

        # Update weights
        # Output layer weights
        self.who += self.lr * np.outer(output_error, hidden_with_bias.ravel())

        # Input layer weights
        self.wih += self.lr * np.outer(hidden_error, x_with_bias.ravel())

        return loss

    def train(self, x: np.ndarray, target: np.ndarray) -> float:
        """
        Train network on single example.

        Args:
            x: Input features [n_inputs]
            target: Target output [n_outputs]

        Returns:
            Loss value
        """
        # Handle both single-layer and multi-layer architectures
        if hasattr(self, "weights") and len(self.weights) > 2:
            # Multi-layer training
            return self._train_multilayer(x, target)
        else:
            # Single-layer training (backward compatibility)
            return self.backward(x, target)

    def _train_multilayer(self, x: np.ndarray, target: np.ndarray) -> float:
        """Multi-layer training with backpropagation."""
        output, activations = self.forward(x)
        target = target.reshape(-1, 1) if target.ndim == 1 else target

        # Calculate output layer error (delta)
        if self.use_softmax:
            # Softmax derivative
            s = output.ravel()
            jacobian = np.diag(s) - np.outer(s, s)
            delta = jacobian @ (target.ravel() - output)
        else:
            # Sigmoid derivative
            delta = (target.ravel() - output) * output * (1 - output)

        deltas = [delta.reshape(-1, 1)]

        # Backpropagate through layers
        for i in reversed(range(len(self.weights) - 1)):
            # Remove bias from weights for backprop
            w_no_bias = self.weights[i + 1][:, :-1]  # Remove bias column
            delta = (
                (w_no_bias.T @ deltas[-1])
                * activations[i + 1][:-1].reshape(-1, 1)
                * (1 - activations[i + 1][:-1].reshape(-1, 1))
            )
            deltas.append(delta)

        deltas.reverse()

        # Update weights
        act_prev = activations[0]  # Input layer
        for i, delta in enumerate(deltas):
            self.weights[i] += self.lr * (delta @ act_prev.reshape(1, -1))
            if i < len(activations) - 1:
                act_prev = activations[i + 1]

        # Calculate loss
        if self.use_softmax:
            loss = cross_entropy_loss(output.reshape(1, -1), target.reshape(1, -1))
        else:
            loss = -np.sum(
                target * np.log(output + 1e-15)
                + (1 - target) * np.log(1 - output + 1e-15)
            )

        return loss

    def predict(self, x: np.ndarray) -> Union[int, np.ndarray]:
        """
        Make prediction - returns class index for classification, raw output for regression.

        Args:
            x: Input features [n_inputs]

        Returns:
            Class index (int) if using softmax, raw output array otherwise
        """
        output = self.get_output(x)  # Always get raw output

        if self.use_softmax:
            return output.argmax()  # Return class with highest probability
        else:
            return output  # Return raw output array for regression/binary tasks

    def get_output(self, x: np.ndarray) -> np.ndarray:
        """
        Get raw network output (before any classification decisions).

        Args:
            x: Input features [n_inputs]

        Returns:
            Raw network output [n_outputs]
        """
        forward_result = self.forward(x)

        # Handle both single-layer (returns output) and multi-layer (returns tuple)
        if isinstance(forward_result, tuple):
            output, _ = forward_result
        else:
            output = forward_result

        return output

    def batch_train(
        self, X: np.ndarray, y: np.ndarray, epochs: int = 100, verbose: bool = False
    ) -> list:
        """
        Train network on batch of examples.

        Args:
            X: Input features [n_samples, n_inputs]
            y: Target outputs [n_samples, n_outputs]
            epochs: Number of training epochs
            verbose: Whether to print progress

        Returns:
            List of average losses per epoch
        """
        losses = []

        for epoch in range(epochs):
            epoch_losses = []

            for x_sample, y_sample in zip(X, y):
                loss = self.train(x_sample, y_sample)
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        return losses

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate network performance.

        Args:
            X: Test inputs [n_samples, n_inputs]
            y: True outputs [n_samples, n_outputs]

        Returns:
            Dictionary with accuracy and loss metrics
        """
        predictions = []
        losses = []

        for x_sample, y_sample in zip(X, y):
            output = self.get_output(x_sample)  # Get raw output for loss calculation
            pred = self.predict(x_sample)  # Get prediction for accuracy calculation
            predictions.append(pred)

            # Calculate loss using raw output
            if self.use_softmax:
                loss = cross_entropy_loss(
                    output.reshape(1, -1), y_sample.reshape(1, -1)
                )
            else:
                loss = -np.sum(
                    y_sample * np.log(output + 1e-15)
                    + (1 - y_sample) * np.log(1 - output + 1e-15)
                )
            losses.append(loss)

        predictions = np.array(predictions)

        # Calculate accuracy
        if self.use_softmax:
            pred_classes = predictions.argmax(axis=1)
            true_classes = y.argmax(axis=1)
        else:
            pred_classes = (predictions > 0.5).astype(int).ravel()
            true_classes = y.astype(int).ravel()

        accuracy = np.mean(pred_classes == true_classes)

        return {
            "accuracy": accuracy,
            "avg_loss": np.mean(losses),
            "predictions": predictions,
        }

    def evaluate_dataset(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Comprehensive evaluation on a dataset using confusion matrix and metrics.

        Args:
            X: Test inputs [n_samples, n_inputs]
            y: True outputs [n_samples, n_outputs] (one-hot for classification)

        Returns:
            Dictionary with accuracy, precision, recall, and confusion matrix
        """
        # Get predictions
        if self.use_softmax:
            # Classification: predict class indices
            preds = np.array([self.predict(x) for x in X])
            targets = y.argmax(axis=1) if y.ndim > 1 else y
            cm = confusion_matrix(preds, targets, self.n_outputs)
        else:
            # Regression/Binary: use raw predictions
            preds = np.array([self.predict(x) for x in X])
            # For binary/regression, create simple confusion matrix
            pred_classes = (preds > 0.5).astype(int).ravel()
            true_classes = (y > 0.5).astype(int).ravel()
            cm = confusion_matrix(pred_classes, true_classes, 2)

        # Calculate metrics
        precision, recall = precision_recall(cm)
        accuracy = (
            np.mean(preds == targets)
            if self.use_softmax
            else np.mean(pred_classes == true_classes)
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            ),
            "confusion_matrix": cm,
        }

    def save_weights(self, filepath: str) -> None:
        """Save network weights to file."""
        weights = {
            "wih": self.wih,
            "who": self.who,
            "config": {
                "n_inputs": self.n_inputs,
                "n_hidden": self.n_hidden,
                "n_outputs": self.n_outputs,
                "lr": self.lr,
                "bias": self.bias,
                "use_softmax": self.use_softmax,
            },
        }
        np.savez(filepath, **weights)

    @classmethod
    def load_weights(cls, filepath: str) -> "NeuralNetwork":
        """Load network from saved weights."""
        # Handle both .npz and bare filenames
        if not filepath.endswith(".npz"):
            filepath += ".npz"
        data = np.load(filepath, allow_pickle=True)
        config = data["config"].item()

        nn = cls(**config)
        nn.wih = data["wih"]
        nn.who = data["who"]

        return nn
