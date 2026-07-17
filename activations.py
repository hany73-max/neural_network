import numpy as np

def sigmoid(z): 
	return 1 / (1 + np.exp(-z))

def sigmoid_derivative(A):
	return A * (1 - A)