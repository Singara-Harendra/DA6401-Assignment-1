"""
Q2.1 - Log 5 sample images per class to a W&B Table
Run this once before starting experiments.
"""

import numpy as np
import wandb
from utils.data_loader import load_data

CLASS_NAMES_MNIST = [
    "Zero", "One", "Two", "Three", "Four",
    "Five", "Six", "Seven", "Eight", "Nine"
]

CLASS_NAMES_FASHION = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def log_sample_images(dataset="mnist"):
    wandb.init(
        project="da6401-mlp",
        name=f"data-explore-{dataset}",
        tags=["exploration", dataset],
    )

    X_train, y_train, _, _, _, _ = load_data(dataset)
    class_names = CLASS_NAMES_MNIST if dataset == "mnist" else CLASS_NAMES_FASHION

    table = wandb.Table(columns=["image", "label", "class_name"])

    for cls in range(10):
        indices = np.where(y_train == cls)[0]
        chosen  = np.random.choice(indices, size=5, replace=False)
        for idx in chosen:
            img = X_train[idx].reshape(28, 28)
            table.add_data(
                wandb.Image(img, caption=f"Class {cls}: {class_names[cls]}"),
                int(cls),
                class_names[cls],
            )

    wandb.log({"sample_images": table})
    wandb.finish()
    print(f"Done — check wandb.ai for the image table.")


if __name__ == "__main__":
    log_sample_images("mnist")