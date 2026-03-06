"""
Inference Script
Evaluate a trained model on the test set
"""

import argparse
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ann.neural_network import NeuralNetwork
from ann.objective_functions import get_loss
from utils.data_loader import load_data


def load_best_config(config_path="best_config.json"):
    """Load saved config to use as default arguments."""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def parse_arguments():
    # load best config as defaults so bare `python inference.py` just works
    cfg = load_best_config()

    parser = argparse.ArgumentParser(description="Run inference with a saved NumPy MLP")

    parser.add_argument("--model_path",          type=str,   default=cfg.get("model_save_path", "best_model.npy"), help="Path to .npy weights file")
    parser.add_argument("-d",  "--dataset",      type=str,   default=cfg.get("dataset",         "mnist"),          help="mnist or fashion_mnist")
    parser.add_argument("-b",  "--batch_size",   type=int,   default=cfg.get("batch_size",       256),             help="Batch size")
    parser.add_argument("-nhl","--num_layers",   type=int,   default=cfg.get("num_layers",       3),               help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size",  type=int,   default=cfg.get("hidden_size",      128),             help="Neurons per hidden layer")
    parser.add_argument("-a",  "--activation",   type=str,   default=cfg.get("activation",       "relu"),          help="relu, sigmoid, tanh")
    parser.add_argument("-wi", "--weight_init",  type=str,   default=cfg.get("weight_init",      "xavier"),        help="xavier, random, zeros")
    parser.add_argument("-l",  "--loss",         type=str,   default=cfg.get("loss",             "cross_entropy"), help="cross_entropy or mse")
    parser.add_argument("-o",  "--optimizer",    type=str,   default=cfg.get("optimizer",        "sgd"),           help="Optimizer name")
    parser.add_argument("-lr", "--learning_rate",type=float, default=cfg.get("learning_rate",    0.01),            help="Learning rate")
    parser.add_argument("-wd", "--weight_decay", type=float, default=cfg.get("weight_decay",     0.0),             help="Weight decay")

    return parser.parse_args()


def load_model(model_path, args):
    """Rebuild network from args and load saved weights."""
    model = NeuralNetwork(args)
    weight_dict = np.load(model_path, allow_pickle=True).item()
    model.set_weights(weight_dict)
    return model


def evaluate_model(model, X_test, y_test, loss_fn):
    """
    Run full evaluation.
    Returns dict: logits, loss, accuracy, precision, recall, f1
    """
    logits = model.forward(X_test)
    preds  = np.argmax(logits, axis=1)
    loss   = loss_fn(logits, y_test)

    return {
        "logits":    logits,
        "loss":      float(loss),
        "accuracy":  float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, average="macro", zero_division=0)),
        "recall":    float(recall_score(y_test, preds,    average="macro", zero_division=0)),
        "f1":        float(f1_score(y_test, preds,        average="macro", zero_division=0)),
    }


def main():
    args = parse_arguments()
    loss_fn, _ = get_loss(args.loss)

    print(f"Loading model : {args.model_path}")
    print(f"Architecture  : {args.num_layers} hidden layers x {args.hidden_size} neurons, activation={args.activation}")
    print(f"Dataset       : {args.dataset}\n")

    _, _, _, _, X_test, y_test = load_data(args.dataset)

    model   = load_model(args.model_path, args)
    results = evaluate_model(model, X_test, y_test, loss_fn)

    print("\n===== Evaluation Results =====")
    print(f"  Loss      : {results['loss']:.4f}")
    print(f"  Accuracy  : {results['accuracy']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"  F1 Score  : {results['f1']:.4f}")
    print("==============================\n")

    print("Evaluation complete!")
    return results


if __name__ == "__main__":
    main()