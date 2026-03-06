"""
Optimization Algorithms
Implements: SGD, Momentum, NAG, RMSProp
"""

import numpy as np


class SGD:
    def __init__(self, lr=0.01, weight_decay=0.0):
        self.lr = lr
        self.weight_decay = weight_decay

    def update(self, layers):
        for layer in layers:
            layer.W -= self.lr * (layer.grad_W + self.weight_decay * layer.W)
            layer.b -= self.lr * layer.grad_b


class Momentum:
    def __init__(self, lr=0.01, beta=0.9, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.vW = None
        self.vb = None

    def _init(self, layers):
        self.vW = [np.zeros_like(l.W) for l in layers]
        self.vb = [np.zeros_like(l.b) for l in layers]

    def update(self, layers):
        if self.vW is None:
            self._init(layers)
        for i, layer in enumerate(layers):
            self.vW[i] = self.beta * self.vW[i] - self.lr * (layer.grad_W + self.weight_decay * layer.W)
            self.vb[i] = self.beta * self.vb[i] - self.lr * layer.grad_b
            layer.W += self.vW[i]
            layer.b += self.vb[i]


class NAG:
    """Nesterov Accelerated Gradient."""

    def __init__(self, lr=0.01, beta=0.9, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.vW = None
        self.vb = None

    def _init(self, layers):
        self.vW = [np.zeros_like(l.W) for l in layers]
        self.vb = [np.zeros_like(l.b) for l in layers]

    def apply_lookahead(self, layers):
        """Shift weights to the Nesterov lookahead position before forward pass."""
        for i, layer in enumerate(layers):
            layer.W += self.beta * self.vW[i]
            layer.b += self.beta * self.vb[i]

    def undo_lookahead(self, layers):
        """Restore original weights after gradient is computed."""
        for i, layer in enumerate(layers):
            layer.W -= self.beta * self.vW[i]
            layer.b -= self.beta * self.vb[i]

    def update(self, layers):
        if self.vW is None:
            self._init(layers)
        for i, layer in enumerate(layers):
            self.vW[i] = self.beta * self.vW[i] - self.lr * (layer.grad_W + self.weight_decay * layer.W)
            self.vb[i] = self.beta * self.vb[i] - self.lr * layer.grad_b
            layer.W += self.vW[i]
            layer.b += self.vb[i]


class RMSProp:
    def __init__(self, lr=0.01, beta=0.9, epsilon=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.sW = None
        self.sb = None

    def _init(self, layers):
        self.sW = [np.zeros_like(l.W) for l in layers]
        self.sb = [np.zeros_like(l.b) for l in layers]

    def update(self, layers):
        if self.sW is None:
            self._init(layers)
        for i, layer in enumerate(layers):
            gW = layer.grad_W + self.weight_decay * layer.W
            gb = layer.grad_b
            self.sW[i] = self.beta * self.sW[i] + (1 - self.beta) * gW ** 2
            self.sb[i] = self.beta * self.sb[i] + (1 - self.beta) * gb ** 2
            layer.W -= self.lr * gW / (np.sqrt(self.sW[i]) + self.epsilon)
            layer.b -= self.lr * gb / (np.sqrt(self.sb[i]) + self.epsilon)


def get_optimizer(name, lr=0.01, weight_decay=0.0):
    """Return an optimizer instance by name."""
    name = name.lower()
    if name == "sgd":
        return SGD(lr=lr, weight_decay=weight_decay)
    elif name == "momentum":
        return Momentum(lr=lr, weight_decay=weight_decay)
    elif name == "nag":
        return NAG(lr=lr, weight_decay=weight_decay)
    elif name == "rmsprop":
        return RMSProp(lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer '{name}'. Choose: sgd, momentum, nag, rmsprop.")