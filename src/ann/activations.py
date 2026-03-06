"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh(z):
    return np.tanh(z)


def tanh_derivative(z):
    return 1.0 - np.tanh(z) ** 2


def softmax(z):
    # subtract row max for numerical stability
    z_stable = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def get_activation(name):
    """Return (forward_fn, derivative_fn) for the given activation name."""
    name = name.lower()
    if name == "relu":
        return relu, relu_derivative
    elif name == "sigmoid":
        return sigmoid, sigmoid_derivative
    elif name == "tanh":
        return tanh, tanh_derivative
    else:
        raise ValueError(f"Unknown activation '{name}'. Choose: relu, sigmoid, tanh.")