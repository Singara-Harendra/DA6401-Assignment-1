"""
Q2.8 - Confusion Matrix for best model
"""

import numpy as np
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork
import json


def plot_confusion_matrix():
    with open("best_config.json", "r") as f:
        cfg = json.load(f)

    wandb.init(project="da6401-mlp", name="confusion-matrix-best-model")

    _, _, _, _, X_test, y_test = load_data(cfg["dataset"])

    class Args:
        pass
    args = Args()
    for k, v in cfg.items():
        setattr(args, k, v)

    model = NeuralNetwork(args)
    weights = np.load("best_model.npy", allow_pickle=True).item()
    model.set_weights(weights)

    logits = model.forward(X_test)
    preds  = np.argmax(logits, axis=1)

    # standard confusion matrix
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=True)
    ax.set_title("Confusion Matrix — Best Model")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)

    # creative failure visualization — show most confused class pairs
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    # zero out diagonal (correct predictions) to show only errors
    cm_errors = cm.copy()
    np.fill_diagonal(cm_errors, 0)
    im = ax2.imshow(cm_errors, cmap="Reds")
    ax2.set_title("Model Failures — Off-diagonal Errors Only")
    ax2.set_xlabel("Predicted Label")
    ax2.set_ylabel("True Label")
    plt.colorbar(im, ax=ax2)
    for i in range(10):
        for j in range(10):
            if cm_errors[i, j] > 0:
                ax2.text(j, i, str(cm_errors[i, j]),
                        ha="center", va="center", fontsize=8,
                        color="black" if cm_errors[i, j] < cm_errors.max()/2 else "white")
    plt.tight_layout()
    plt.savefig("confusion_matrix_errors.png", dpi=150)

    wandb.log({
        "confusion_matrix":        wandb.Image("confusion_matrix.png"),
        "confusion_matrix_errors": wandb.Image("confusion_matrix_errors.png"),
    })
    wandb.finish()
    print("Done — both plots logged to W&B.")


if __name__ == "__main__":
    plot_confusion_matrix()