import src.data_generation as data_generation
import src.dense_layer as dense_layer
import src.neural_network as neural_network
import src.visualizaiton as visualization
import numpy as np
import pandas as pd

lr = .001
n_epochs = 1000
n_layers = 3
n_neurons = [1000]
VALID_HIDDEN_ACTIVATIONS = ["relu", "sigmoid", "tanh"]
VALID_OUTPUT_ACTIVATIONS = ["sigmoid", "linear"]

n_features = data_generation.X.shape[1] 
n_outputs = data_generation.y.shape[1]

def build_network(n_layers, n_features, n_outputs, n_neurons):
    model = neural_network.NeuralNetwork()

    for i in range(n_layers):
        if i == 0:
            model.add_layer(dense_layer.DenseLayer(n_features, n_neurons, "tanh"))
        elif i > 0 and i < n_layers-1:
            model.add_layer(dense_layer.DenseLayer(n_neurons, n_neurons, "tanh"))
        else:
            model.add_layer(dense_layer.DenseLayer(n_neurons, n_outputs, "linear"))    

    return model        

def train(X, y, model):
    A_finals = []
    layer_snapshots = []
    costs = []
    for epoch in range(n_epochs):
        print(f"epoch {epoch} of {n_epochs}")
        A_final = model.forward(X)
        cost = model.cost_calc(A_final, y)
        costs.append(cost)

        if epoch % 50 == 0:
            A_finals.append((epoch, A_final.copy()))
            layer_snapshots.append((epoch, [layer.A.copy() for layer in model.layers]))

        model.backward(A_final, y)
        model.update(lr)
    return A_finals, layer_snapshots, costs

if __name__ == "__main__":
    for i in n_neurons:
        model = build_network(n_layers, n_features, n_outputs, i)
        A_finals, layer_snapshots, costs = train(data_generation.X_scaled, data_generation.y, model)
        visualization.visualize_training_run(
            data_generation.X, data_generation.y, A_finals, layer_snapshots, costs
        )

