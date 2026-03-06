"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np
from ann.activations import get_activation


class Layer:
    """
    A single fully-connected (dense) layer.

    After forward()  -> self.z (pre-activation), self.a (post-activation) are cached.
    After backward() -> self.grad_W and self.grad_b are populated (required by autograder).
    """

    def __init__(self, input_size, output_size, activation="relu", weight_init="xavier"):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation

        self.act_fn, self.act_deriv = get_activation(activation)

        # weight initialisation
        if weight_init == "xavier":
            limit = np.sqrt(6.0 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        elif weight_init == "random":
            self.W = np.random.randn(input_size, output_size) * 0.01
        elif weight_init == "zeros":
            self.W = np.zeros((input_size, output_size))
        else:
            raise ValueError(f"Unknown weight_init '{weight_init}'. Choose: xavier, random, zeros.")

        self.b = np.zeros((1, output_size))

        # gradients — exposed after every backward() call (autograder checks these)
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        # forward cache
        self.input = None
        self.z = None
        self.a = None

    def forward(self, x):
        """
        x   : (batch_size, input_size)
        returns a : (batch_size, output_size)
        """
        self.input = x
        self.z = x @ self.W + self.b      # pre-activation
        self.a = self.act_fn(self.z)      # post-activation
        return self.a

    def backward(self, delta):
        """
        delta : upstream gradient w.r.t. this layer's post-activation output
                shape (batch_size, output_size)

        Computes grad_W, grad_b and returns gradient for the previous layer.
        """
        batch_size = self.input.shape[0]

        # multiply upstream gradient by local activation derivative -> dL/dz
        dz = delta * self.act_deriv(self.z)

        # parameter gradients (averaged over batch)
        self.grad_W = (self.input.T @ dz) / batch_size
        self.grad_b = np.sum(dz, axis=0, keepdims=True) / batch_size

        # gradient to pass to previous layer
        return dz @ self.W.T