import numpy as np

a = np.eye(2)
print(np.concatenate((np.zeros((2, 4)), a), axis=1))
