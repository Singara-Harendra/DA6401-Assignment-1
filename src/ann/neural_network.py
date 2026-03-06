"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

import numpy as np
from ann.neural_layer import Layer
from ann.objective_functions import get_loss
from ann.optimizers import get_optimizer, NAG


class NeuralNetwork:
    """
    Configurable Multi-Layer Perceptron built entirely with NumPy.
    Supports variable depth with different neuron counts per layer.
    """

    def __init__(self, cli_args):
        self.args = cli_args

        input_size  = 784
        output_size = 10
        num_layers  = cli_args.num_layers
        activation  = cli_args.activation
        weight_init = cli_args.weight_init

        # hidden_size can be int or list — handle both
        hidden_size = cli_args.hidden_size
        if isinstance(hidden_size, int):
            hidden_sizes = [hidden_size] * num_layers
        elif isinstance(hidden_size, list):
            if len(hidden_size) == 1:
                hidden_sizes = hidden_size * num_layers
            elif len(hidden_size) == num_layers:
                hidden_sizes = hidden_size
            else:
                # fill missing entries with last value
                hidden_sizes = hidden_size + [hidden_size[-1]] * (num_layers - len(hidden_size))
        else:
            hidden_sizes = [128] * num_layers

        # build layers: input -> hidden... -> output
        sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = []
        for i in range(len(sizes) - 1):
            is_output = (i == len(sizes) - 2)
            act = activation if not is_output else "relu"
            self.layers.append(
                Layer(sizes[i], sizes[i + 1], activation=act, weight_init=weight_init)
            )

        # output layer is linear — returns raw logits, softmax applied in loss
        self.layers[-1].act_fn    = lambda z: z
        self.layers[-1].act_deriv = lambda z: np.ones_like(z)

        self.loss_fn, self.loss_grad = get_loss(cli_args.loss)
        self.optimizer = get_optimizer(
            cli_args.optimizer,
            lr=cli_args.learning_rate,
            weight_decay=cli_args.weight_decay,
        )

        if isinstance(self.optimizer, NAG):
            self.optimizer._init(self.layers)

        self.grad_W = None
        self.grad_b = None

    def forward(self, X):
        """
        X : (batch, 784)
        Returns logits : (batch, 10) — softmax NOT applied here.
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, y_true, y_pred):
        """
        Backpropagation through all layers.
        grad_W[0]/grad_b[0] = output layer (last layer).
        grad_W[-1]/grad_b[-1] = first hidden layer.
        """
        grad_W_list = []
        grad_b_list = []

        delta = self.loss_grad(y_pred, y_true)

        for layer in reversed(self.layers):
            delta = layer.backward(delta)
            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

    def update_weights(self):
        self.optimizer.update(self.layers)

    def get_gradient_norms(self):
        """L2 norm of grad_W per layer. index 0 = first layer."""
        return [float(np.linalg.norm(layer.grad_W)) for layer in self.layers]

    def get_activation_stats(self):
        """Dead neuron fraction and mean activation per hidden layer."""
        stats = []
        for layer in self.layers[:-1]:
            if layer.a is not None:
                stats.append({
                    "dead_frac": float(np.mean(layer.a == 0)),
                    "mean_act":  float(np.mean(np.abs(layer.a))),
                })
        return stats

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=1, batch_size=32, wandb_log=False):
        """Mini-batch training loop. Returns (history, best_weights)."""
        import wandb

        n = X_train.shape[0]
        history      = []
        best_val_acc = -1.0
        best_weights = self.get_weights()

        for epoch in range(epochs):
            perm = np.random.permutation(n)
            X_s, y_s = X_train[perm], y_train[perm]

            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, n, batch_size):
                Xb = X_s[start: start + batch_size]
                yb = y_s[start: start + batch_size]

                if isinstance(self.optimizer, NAG):
                    self.optimizer.apply_lookahead(self.layers)
                    logits = self.forward(Xb)
                    loss   = self.loss_fn(logits, yb)
                    self.optimizer.undo_lookahead(self.layers)
                    logits = self.forward(Xb)
                else:
                    logits = self.forward(Xb)
                    loss   = self.loss_fn(logits, yb)

                self.backward(yb, logits)
                self.update_weights()

                epoch_loss += loss
                n_batches  += 1

            avg_loss  = epoch_loss / n_batches
            train_acc = self.evaluate(X_train, y_train)
            val_acc   = self.evaluate(X_val, y_val) if X_val is not None else 0.0

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = self.get_weights()

            history.append({
                "epoch": epoch + 1, "loss": avg_loss,
                "train_acc": train_acc, "val_acc": val_acc,
            })
            print(f"Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  "
                  f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

            if wandb_log:
                log_dict = {
                    "epoch":          epoch + 1,
                    "train_loss":     avg_loss,
                    "train_accuracy": train_acc,
                    "val_accuracy":   val_acc,
                }
                for li, norm in enumerate(self.get_gradient_norms()):
                    log_dict[f"grad_norm_layer_{li}"] = norm
                for li, stat in enumerate(self.get_activation_stats()):
                    log_dict[f"dead_frac_layer_{li}"] = stat["dead_frac"]
                    log_dict[f"mean_act_layer_{li}"]  = stat["mean_act"]
                wandb.log(log_dict)

        return history, best_weights

    def evaluate(self, X, y):
        """Return classification accuracy."""
        preds = np.argmax(self.forward(X), axis=1)
        return float(np.mean(preds == y))

    def get_weights(self):
        d = {}
        for i, l in enumerate(self.layers):
            d[f"W{i}"] = l.W.copy()
            d[f"b{i}"] = l.b.copy()
        return d

    def set_weights(self, d):
        for i, layer in enumerate(self.layers):
            if f"W{i}" in d: layer.W = d[f"W{i}"].copy()
            if f"b{i}" in d: layer.b = d[f"b{i}"].copy()