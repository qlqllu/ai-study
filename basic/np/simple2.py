import numpy as np
import math

n = 10
v = np.random.random(n)
u = np.zeros(n)

for i in range(n):
  u[i] = math.exp(v[i])

print(v)
print(u)
