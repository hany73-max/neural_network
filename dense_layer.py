import numpy as np
from activations import sigmoid , sigmoid_derivative

class DenseLayer:
    # the input dim is the (n) and the output is (h). for notations look into the forward pass doc.
    def __init__(self, input_dim, output_dim, activation_function):
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))
        self.activation_function = activation_function.lower()

        self.dZ = None
        self.dW = None
        self.db = None

    def forward(self, A_prev):
        self.A_prev = A_prev
        self.Z = np.dot(self.A_prev, self.W) + self.b

        if self.activation_function == "relu":
            self.A = np.maximum(0, self.Z)
        elif self.activation_function == "sigmoid":
            self.A = sigmoid(self.Z)

        return self.A


    def backward(self, dA):

        if self.activation_function == "relu":
            self.dZ = dA * (self.Z > 0)
        elif self.activation_function == "sigmoid":
            self.dZ = dA * (sigmoid_derivative(self.A)) 

        self.dW = np.dot(self.A_prev.T, self.dZ) 

        self.db = np.sum(self.dZ, axis=0, keepdims=True) 

        dA_prev = np.dot(self.dZ, self.W.T)

        return dA_prev

    def update(self, learning_rate):
        self.W = self.W - learning_rate*self.dW
        self.b = self.b - learning_rate*self.db