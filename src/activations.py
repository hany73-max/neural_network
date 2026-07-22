import numpy as np

def sigmoid(z): 
	return 1 / (1 + np.exp(-z))

def sigmoid_derivative(A):
	return A * (1 - A)

def ReLU(Z):
	return np.maximum(0, Z)

def ReLU_derivative(Z):
	return Z > 0

def linear(Z):
	return Z

def linear_derivative(Z):
	return 1

def tanh(Z):
	return np.tanh(Z)

def tanh_derivative(Z):
	return 1 - (np.tanh(Z) ** 2)