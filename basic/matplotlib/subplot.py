import matplotlib.pyplot as plt
import numpy as np

f, plts = plt.subplots(3, 3, sharex=True, sharey=False)
print(plts.shape)

x = np.array(range(20))
for i, p in enumerate(plts.flat):
  y = x ** i
  p.plot(x, y)
  
plt.show()