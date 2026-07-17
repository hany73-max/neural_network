import data_generation
import dense_layer
import neural_network
import numpy as np

lr = .01
n_epochs = 10
n_layers = 2
n_neurons = [10, 100 , 1000]

n_features = data_generation.X.shape[1] 
n_outputs = data_generation.y.shape[1]

def build_network(n_layers, n_features, n_outputs, n_neurons):
    model = neural_network.NeuralNetwork()

    for i in range(n_layers):
        if i == 0:
            model.add_layer(dense_layer.DenseLayer(n_features, n_neurons, "relu"))
        elif i > 0 and i < n_layers-1:
            model.add_layer(dense_layer.DenseLayer(n_neurons, n_neurons, "relu"))
        else:
            model.add_layer(dense_layer.DenseLayer(n_neurons, n_outputs, "linear"))    

    return model        

def train(X, y, model):
    for epoch in range(n_epochs):
        A_final = model.forward(X)
        cost = model.cost_calc(A_final, y)
        print(f"Epoch {epoch+1}/{n_epochs} - Cost: {cost}")
        model.backward(A_final,y)
        model.update(lr)

if __name__ == "__main__":
    for i in n_neurons:
        model = build_network(n_layers, n_features, n_outputs, i)
        train(data_generation.X, data_generation.y, model)