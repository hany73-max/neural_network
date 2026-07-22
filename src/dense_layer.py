import numpy as np
from src.activations import sigmoid , sigmoid_derivative, ReLU, ReLU_derivative, linear, linear_derivative , tanh , tanh_derivative

class DenseLayer:
    # the input dim is the (n) and the output is (h). for notations look into the forward pass doc.
    def __init__(self, input_dim, output_dim, activation_function):
        self.activation_function = activation_function.lower()

        if self.activation_function == "relu":
            scale = np.sqrt(2 / input_dim)      # He init
        else:
            scale = np.sqrt(1 / input_dim)      # Xavier init (tanh/sigmoid/linear)

        self.W = np.random.randn(input_dim, output_dim) * scale
        self.b = np.zeros((1, output_dim))

        self.dZ = None
        self.dW = None
        self.db = None

    def forward(self, A_prev):
        self.A_prev = A_prev
        self.Z = np.dot(self.A_prev, self.W) + self.b

        if self.activation_function == "relu":
            self.A = ReLU(self.Z)
        elif self.activation_function == "sigmoid":
            self.A = sigmoid(self.Z)
        elif self.activation_function == "linear":
            self.A = linear(self.Z)
        elif self.activation_function == "tanh":
            self.A = tanh(self.Z)
        return self.A


    def backward(self, dA):

        if self.activation_function == "relu":
            self.dZ = dA * (ReLU_derivative(self.Z))
        elif self.activation_function == "sigmoid":
            self.dZ = dA * (sigmoid_derivative(self.A)) 
        elif self.activation_function == "linear":
            self.dZ = dA * (linear_derivative(self.A)) 
        elif self.activation_function == "tanh":
            self.dZ = dA * (tanh_derivative(self.Z))

        self.dW = np.dot(self.A_prev.T, self.dZ) 

        self.db = np.sum(self.dZ, axis=0, keepdims=True) 

        dA_prev = np.dot(self.dZ, self.W.T)

        return dA_prev

    def update(self, learning_rate):
        self.W = self.W - learning_rate*self.dW
        self.b = self.b - learning_rate*self.db