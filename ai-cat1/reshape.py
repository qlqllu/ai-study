"""
read h5 file and write image
"""
import os
import h5py
import numpy as np

dir_path = os.path.dirname(__file__)

f = h5py.File('train_cat.h5', 'r')
datasetNames = [n for n in f.keys()]
for n in datasetNames:
  print(n, f[n].shape)

x1 = np.array(f['train_set_x'][0])
print(x1.shape)

x1 = x1.reshape(x1.shape[0] * x1.shape[1] * x1.shape[2])
print(x1.shape)

f.close()
