import numpy as np

# print(np.linspace(0, 2 * np.pi, 400))

# a1 = np.array([1, 2, 3])
# print(a1.shape)
# a1.reshape(3)
# print(a1.shape)

# a2 = np.array((1, 2, 3))
# print(a2)

a3 = np.array([1, 2, 3])
b = 2
c = np.array([1, 2, 3])
print(a3 * b)
print(np.multiply(a3, b))
print(a3 * c)
print(np.multiply(a3, c))
print(np.dot(a3, c))
