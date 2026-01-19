"""
ML Foundations Utilities

Core utility functions for neural network implementation:
- Data preprocessing and scaling
- Weight initialization
- Evaluation metrics
- Activation functions
"""

import numpy as np
from scipy.stats import truncnorm
from typing import Tuple, Union, Optional


def truncated_normal(
    mean: float = 0,
    sd: float = 1,
    low: float = -0.5,
    upp: float = 0.5
) -> truncnorm:
    """
    Create a truncated normal distribution for weight initialization.

    Prevents saturation by limiting weight range. Used for Xavier/Glorot initialization.

    Args:
        mean: Distribution mean (default: 0)
        sd: Standard deviation (default: 1)
        low: Lower bound (default: -0.5)
        upp: Upper bound (default: 0.5)

    Returns:
        Truncated normal distribution object
    """
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def scale_to_01(data: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
    """
    Scale data to [epsilon, 1-epsilon] range to avoid zero-input stagnation.

    Neural networks struggle with exact zeros in inputs/weights as gradients
    become zero. This scaling ensures all values contribute to learning.

    Args:
        data: Input data array (typically 0-255 for images)
        epsilon: Small offset to avoid exact 0/1 (default: 0.01)

    Returns:
        Scaled data in [epsilon, 1-epsilon] range
    """
    if data.max() == data.min():
        return np.full_like(data, 0.5, dtype=float)  # Handle constant arrays

    # Scale to [0, 1] first
    scaled = (data - data.min()) / (data.max() - data.min())

    # Then map to [epsilon, 1-epsilon]
    return scaled * (1 - 2 * epsilon) + epsilon


def one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Convert class labels to one-hot encoding with epsilon adjustment.

    Avoids exact 0s which can stall learning.

    Args:
        labels: Integer class labels
        n_classes: Number of classes

    Returns:
        One-hot encoded array with values in [0.01, 0.99]
    """
    oh = (np.arange(n_classes) == labels.reshape(-1, 1)).astype(float)
    oh[oh == 0] = 0.01  # Avoid exact zeros
    oh[oh == 1] = 0.99  # Avoid exact ones
    return oh


def confusion_matrix(
    preds: np.ndarray,
    targets: np.ndarray,
    n_classes: Optional[int] = None
) -> np.ndarray:
    """
    Compute confusion matrix for multi-class classification.

    Standard convention: rows = true labels, columns = predicted labels

    Args:
        preds: Predicted class indices or one-hot arrays
        targets: True class indices or one-hot arrays
        n_classes: Number of classes (inferred if None)

    Returns:
        Confusion matrix [n_classes x n_classes]
    """
    # Handle one-hot inputs
    if preds.ndim > 1 and preds.shape[1] > 1:
        preds = preds.argmax(axis=1)
    if targets.ndim > 1 and targets.shape[1] > 1:
        targets = targets.argmax(axis=1)

    if n_classes is None:
        n_classes = max(preds.max(), targets.max()) + 1

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for pred, target in zip(preds, targets):
        cm[int(target), int(pred)] += 1  # Note: target first, then pred
    return cm


def precision_recall(cm: np.ndarray) -> Tuple[float, float]:
    """
    Calculate macro-averaged precision and recall from confusion matrix.

    Args:
        cm: Confusion matrix [n_classes x n_classes]

    Returns:
        Tuple of (macro_precision, macro_recall)
    """
    n_classes = cm.shape[0]
    precisions = []
    recalls = []

    for i in range(n_classes):
        # True positives for class i
        tp = cm[i, i]

        # Predicted positives for class i
        predicted_pos = cm[i, :].sum()

        # Actual positives for class i
        actual_pos = cm[:, i].sum()

        # Precision: TP / (TP + FP)
        precision = tp / predicted_pos if predicted_pos > 0 else 0.0
        precisions.append(precision)

        # Recall: TP / (TP + FN)
        recall = tp / actual_pos if actual_pos > 0 else 0.0
        recalls.append(recall)

    return np.mean(precisions), np.mean(recalls)


def accuracy_score(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate accuracy score.

    Args:
        preds: Predictions (indices or one-hot)
        targets: True labels (indices or one-hot)

    Returns:
        Accuracy as float [0, 1]
    """
    # Handle one-hot inputs
    if preds.ndim > 1 and preds.shape[1] > 1:
        preds = preds.argmax(axis=1)
    if targets.ndim > 1 and targets.shape[1] > 1:
        targets = targets.argmax(axis=1)

    return np.mean(preds == targets)


@np.vectorize
def sigmoid(x: float) -> float:
    """
    Sigmoid activation function with overflow protection.

    Maps any real number to (0, 1) range. Derivative: sigmoid(x) * (1 - sigmoid(x))

    Args:
        x: Input value

    Returns:
        Sigmoid output in (0, 1)
    """
    # Prevent overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax activation function with numerical stability.

    Converts logits to probabilities that sum to 1.

    Args:
        x: Input logits
        axis: Axis along which to compute softmax

    Returns:
        Probability distribution
    """
    # Subtract max for numerical stability
    x_shifted = x - x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def cross_entropy_loss(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate cross-entropy loss for multi-class classification.

    Args:
        preds: Predicted probabilities [batch_size, n_classes]
        targets: True labels (one-hot encoded) [batch_size, n_classes]

    Returns:
        Average cross-entropy loss
    """
    # Avoid log(0) by clipping predictions
    preds = np.clip(preds, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(targets * np.log(preds), axis=1))


def binary_cross_entropy_loss(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate binary cross-entropy loss.

    Args:
        preds: Predicted probabilities [batch_size]
        targets: True labels [batch_size]

    Returns:
        Average binary cross-entropy loss
    """
    preds = np.clip(preds, 1e-15, 1 - 1e-15)
    return -np.mean(targets * np.log(preds) + (1 - targets) * np.log(1 - preds))