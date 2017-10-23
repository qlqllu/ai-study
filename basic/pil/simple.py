from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open('test.png')
img = np.array(img)
plt.imshow(img)
plt.show()
