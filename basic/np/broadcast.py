import numpy as np

## n = 3, m = 2
b = 0
w = np.array([1, 2, 3]).reshape(3, 1)
x = np.array([[1, 2],
  [3, 4],
  [5, 6]]
)

z = np.dot(w.T, x) + b
print(x.shape)
print(z)
