import numpy as np

def Standardization(X):
    return (X - X.mean())/X.std()

end = 100
X = range(0, end)
X = np.array(X).reshape(-1, 1)
y = np.sin(X) + ((1/2) *np.sin(2*X)) + ((1/3)* np.sin(3*X))
X_scaled = Standardization(X)