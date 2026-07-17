import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        current_input = X 
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input

    def cost_calc(self, A_final, y):
        squared_error = (y - A_final) ** 2
        cost = np.mean(np.sum(squared_error, axis=1))
        return cost

    def backward(self, A_final, y):
        m = y.shape[0]
        dA = (2 / m) * (A_final - y)

        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)