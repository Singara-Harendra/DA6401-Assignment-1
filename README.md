# DA6401 Assignment 1 — Multi-Layer Perceptron from Scratch

## Name    : Singara Harendra
## Roll No : DA25M028

## Overview
Implementation of a configurable Multi-Layer Perceptron (MLP) using only NumPy to classify MNIST and Fashion-MNIST datasets. Implements forward propagation, backpropagation, and multiple optimization strategies from scratch.

## W&B Report
https://wandb.ai/da25m028-indian-institute-of-technology-madras/da6401-mlp/reports/DA6401-Assignment-1-MLP-from-Scratch--VmlldzoxNjEyNTQyMA?accessToken=ge1p1w4rm6ndmg2ajv9ed9exf14ly8r0lycp1oxmvlxa1kbgbotzfa9nhotu37fn

## GitHub Repository
https://github.com/Singara-Harendra/DA6401-Assignment-1

## W&B Project
https://wandb.ai/da25m028-indian-institute-of-technology-madras/da6401-mlp?nw=nwuserda25m028

## Setup
```bash
pip install numpy matplotlib keras scikit-learn wandb tensorflow
wandb login
```

## Training
```bash
cd src
python train.py -d mnist -e 10 -b 64 -o momentum -lr 0.01 -nhl 3 -sz 128 -a relu -wi xavier -l cross_entropy
```

## Inference
```bash
cd src
python inference.py
```

## Arguments
| Flag | Description | Options |
|------|-------------|---------|
| -d | Dataset | mnist, fashion_mnist |
| -e | Epochs | integer |
| -b | Batch size | integer |
| -l | Loss function | cross_entropy, mse |
| -o | Optimizer | sgd, momentum, nag, rmsprop |
| -lr | Learning rate | float |
| -wd | Weight decay | float |
| -nhl | Number of hidden layers | integer |
| -sz | Neurons per hidden layer | integer(s) e.g. 128 or 128 64 |
| -a | Activation | relu, sigmoid, tanh |
| -wi | Weight init | xavier, random, zeros |

## Project Structure
```
da6401_assignment_1/
├── README.md
├── requirements.txt
├── sweep_config.yaml
└── src/
    ├── train.py
    ├── inference.py
    ├── explore_data.py
    ├── confusion_matrix.py
    ├── best_model.npy
    ├── best_config.json
    ├── ann/
    │   ├── activations.py
    │   ├── neural_layer.py
    │   ├── neural_network.py
    │   ├── optimizers.py
    │   └── objective_functions.py
    └── utils/
        └── data_loader.py
```

## Results
- Best Test Accuracy: 97.63%
- Best Test F1 Score: 0.9742
