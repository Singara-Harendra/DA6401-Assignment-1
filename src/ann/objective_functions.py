"""
Loss / Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np
from ann.activations import softmax


def cross_entropy_loss(logits, y_true):
    """
    logits : (batch, num_classes)  — raw pre-softmax values
    y_true : (batch,)              — integer class labels
    Returns scalar loss.
    """
    probs = softmax(logits)
    batch_size = y_true.shape[0]
    correct_probs = probs[np.arange(batch_size), y_true]
    return float(-np.mean(np.log(correct_probs + 1e-12)))


def cross_entropy_gradient(logits, y_true):
    """
    Gradient of cross-entropy loss w.r.t. logits (analytically combined with softmax).
    Returns (batch, num_classes).
    """
    probs = softmax(logits)
    batch_size = y_true.shape[0]
    probs[np.arange(batch_size), y_true] -= 1.0
    return probs / batch_size


def mse_loss(logits, y_true, num_classes=10):
    """
    MSE loss: applies softmax to logits first, then computes MSE vs one-hot targets.
    """
    probs = softmax(logits)
    batch_size = y_true.shape[0]
    y_onehot = np.zeros((batch_size, num_classes))
    y_onehot[np.arange(batch_size), y_true] = 1.0
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def mse_gradient(logits, y_true, num_classes=10):
    """
    Gradient of MSE loss w.r.t. logits (through the softmax).
    """
    probs = softmax(logits)
    batch_size = y_true.shape[0]
    y_onehot = np.zeros((batch_size, num_classes))
    y_onehot[np.arange(batch_size), y_true] = 1.0

    diff = probs - y_onehot
    # derivative through softmax: dL/dz_i = sum_j (2 * diff_j * p_j * (delta_ij - p_i))
    sdp = np.sum(diff * probs, axis=1, keepdims=True)
    return 2.0 * probs * (diff - sdp) / batch_size


def get_loss(name):
    """Return (loss_fn, gradient_fn) for the given loss name."""
    name = name.lower()
    if name in ("cross_entropy", "ce"):
        return cross_entropy_loss, cross_entropy_gradient
    elif name in ("mse", "mean_squared_error"):
        return mse_loss, mse_gradient
    else:
        raise ValueError(f"Unknown loss '{name}'. Choose: cross_entropy, mse.")