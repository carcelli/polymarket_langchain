"""
ML Foundations Package

Core neural network building blocks for agent-driven ML solutions.
Provides reusable components for classification, evaluation, and feature processing.

Key Components:
- NeuralNetwork: Multi-layer perceptron with backpropagation
- Utilities: Data preprocessing, evaluation metrics, weight initialization
- Integration: Connects with Polymarket agents for crowd-sourced features
"""

from .nn import NeuralNetwork
from .utils import (
    truncated_normal,
    scale_to_01,
    one_hot,
    confusion_matrix,
    precision_recall,
    sigmoid,
    softmax,
    cross_entropy_loss,
    binary_cross_entropy_loss
)

__all__ = [
    'NeuralNetwork',
    'truncated_normal',
    'scale_to_01',
    'one_hot',
    'confusion_matrix',
    'precision_recall',
    'sigmoid',
    'softmax'
]