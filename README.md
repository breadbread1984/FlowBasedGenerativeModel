# FlowBasedGenerativeModel
This project is a clean refactor of the project [here](https://github.com/ericjang/normalizing-flows-tutorial). The project demos the usage of tensorflow's tfp.bijectors api in implementing a simple flow-based generative model.

## Introduction
This demo shows how a flow-based generative model which is composed of only linear (affine) and non-linear (ReLU) transforms is trained. The trained model can generate samples following a predefined multivariate normal distribution.

## Train the model
start the training with the following command

```bash
python3 train_model.py
``` 

