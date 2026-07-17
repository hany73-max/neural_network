import numpy as np


end = 100
X = range(0, end)
X = np.array(X).reshape(-1, 1)
y = np.sin(X) + ((1/2) *np.sin(2*X)) + ((1/3)* np.sin(3*X))
