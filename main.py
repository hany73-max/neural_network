import data_generation
import dense_layer
import neural_network

lr = .01
n_epochs = 10
n_layers = 2
n_neurons = [10, 100 , 1000]

n_features = data_generation.X.shape[1] 
n_outputs = data_generation.y.shape[1]

def build_network(n_layers, n_features, n_outputs, n_neurons):
    for i in range(n_layers):
        if i == 0:
            dense_layer.DenseLayer.add_layer(dense_layer.DenseLayer(n_features, n_neurons, "relu"))
        elif i > 0 and i < n_layers:
            dense_layer.DenseLayer.add_layer(dense_layer.DenseLayer(n_neurons, n_neurons, "relu"))
        else:
            dense_layer.DenseLayer.add_layer(dense_layer.DenseLayer(n_neurons, n_outputs, "sigmoid"))            

def train(X, y):
    for epoch in range(n_epochs):
        A_final = neural_network.NeuralNetwork.forward(X)
        cost = neural_network.NeuralNetwork.cost_calc(A_final, y)
        print(f"Epoch {epoch+1}/{epoch} - Cost: {cost}")
        neural_network.NeuralNetwork.backward(A_final,y)
        neural_network.NeuralNetwork.update(lr)

if __name__ == "__main__":
    for i in n_neurons:
        build_network(n_layers, n_features, n_outputs, n_neurons)
        train(data_generation.X, data_generation.y)