"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import json
import numpy as np
import wandb
from sklearn.metrics import f1_score

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a NumPy MLP on MNIST / Fashion-MNIST")

    parser.add_argument("-d",   "--dataset",       type=str,   default="mnist",          help="mnist or fashion_mnist")
    parser.add_argument("-e",   "--epochs",         type=int,   default=10,               help="Number of training epochs")
    parser.add_argument("-b",   "--batch_size",     type=int,   default=32,               help="Mini-batch size")
    parser.add_argument("-l",   "--loss",           type=str,   default="cross_entropy",  help="cross_entropy or mse")
    parser.add_argument("-o",   "--optimizer",      type=str,   default="sgd",            help="sgd, momentum, nag, rmsprop")
    parser.add_argument("-lr",  "--learning_rate",  type=float, default=0.01,             help="Learning rate")
    parser.add_argument("-wd",  "--weight_decay",   type=float, default=0.0,              help="L2 weight decay")
    parser.add_argument("-nhl", "--num_layers",     type=int,   default=3,                help="Number of hidden layers")
    parser.add_argument("-sz",  "--hidden_size",    type=int,   default=[128], nargs='+', help="Neurons per hidden layer e.g. -sz 128 64 32")
    parser.add_argument("-a",   "--activation",     type=str,   default="relu",           help="relu, sigmoid, tanh")
    parser.add_argument("-wi",  "--weight_init",    type=str,   default="xavier",         help="xavier, random, zeros")
    parser.add_argument("--wandb_project",          type=str,   default="da6401-mlp",     help="W&B project name")
    parser.add_argument("--wandb_entity",           type=str,   default=None,             help="W&B username")
    parser.add_argument("--model_save_path",        type=str,   default="best_model.npy", help="Path to save weights")
    parser.add_argument("--no_wandb",               action="store_true",                  help="Disable W&B logging")

    return parser.parse_args()


def compute_f1(model, X, y):
    """Compute macro F1 score on given data."""
    logits = model.forward(X)
    preds  = np.argmax(logits, axis=1)
    return float(f1_score(y, preds, average="macro", zero_division=0))


def main():
    args = parse_arguments()
    use_wandb = not args.no_wandb

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
        )
        # allow sweep to override args
        for key, val in wandb.config.items():
            if hasattr(args, key):
                setattr(args, key, val)

    # load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.dataset)

    # build and train model
    model = NeuralNetwork(args)
    history, best_weights = model.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        wandb_log=use_wandb,
    )

    # evaluate best weights from this run on test set
    model.set_weights(best_weights)
    test_acc = model.evaluate(X_test, y_test)
    test_f1  = compute_f1(model, X_test, y_test)

    print(f"\nTest Accuracy : {test_acc:.4f}")
    print(f"Test F1 Score : {test_f1:.4f}")

    if use_wandb:
        wandb.log({"test_accuracy": test_acc, "test_f1": test_f1})
        wandb.finish()

    # ----------------------------------------------------------------
    # save only if this run beats the previous best F1 score
    # fixed filenames: best_model.npy and best_config.json
    # ----------------------------------------------------------------
    prev_best_f1 = 0.0
    try:
        with open("best_config.json", "r") as f:
            prev_best_f1 = json.load(f).get("test_f1", 0.0)
    except FileNotFoundError:
        prev_best_f1 = 0.0

    if test_f1 > prev_best_f1:
        # save weights
        np.save("best_model.npy", best_weights)
        print(f"\nNew best model saved → best_model.npy")
        print(f"  F1 improved: {prev_best_f1:.4f} → {test_f1:.4f}")

        # save config with metrics recorded inside
        config_to_save = vars(args).copy()
        config_to_save["test_accuracy"] = round(float(test_acc), 6)
        config_to_save["test_f1"]       = round(float(test_f1), 6)
        with open("best_config.json", "w") as f:
            json.dump(config_to_save, f, indent=2)
        print(f"Config  saved → best_config.json")
    else:
        print(f"\nModel NOT saved — current F1 {test_f1:.4f} did not beat saved best {prev_best_f1:.4f}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()