"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import numpy as np


def load_data(dataset="mnist", val_split=0.1):
    """
    Load MNIST or Fashion-MNIST via keras.datasets.
    Normalises pixels to [0,1], flattens images to (N, 784).
    Returns: X_train, y_train, X_val, y_val, X_test, y_test
    """
    if dataset == "mnist":
        from keras.datasets import mnist
        (X_full, y_full), (X_test, y_test) = mnist.load_data()
    elif dataset == "fashion_mnist":
        from keras.datasets import fashion_mnist
        (X_full, y_full), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose: mnist, fashion_mnist.")

    # normalise and flatten
    X_full = X_full.reshape(-1, 784).astype(np.float32) / 255.0
    X_test = X_test.reshape(-1, 784).astype(np.float32) / 255.0

    # train / val split
    n_val = int(len(X_full) * val_split)
    idx   = np.random.permutation(len(X_full))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train, y_train = X_full[train_idx], y_full[train_idx]
    X_val,   y_val   = X_full[val_idx],   y_full[val_idx]

    print(f"Loaded {dataset}: train={X_train.shape[0]}  val={X_val.shape[0]}  test={X_test.shape[0]}")
    return X_train, y_train, X_val, y_val, X_test, y_test