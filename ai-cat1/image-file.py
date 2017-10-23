"""
read h5 file and write image
"""
import os
import h5py
from PIL import Image

dir_path = os.path.dirname(__file__)

f = h5py.File('train_cat.h5', 'r')
datasetNames = [n for n in f.keys()]
for n in datasetNames:
  print(n, f[n].shape)

for i, data in enumerate(f['train_set_x']):
  img = Image.fromarray(data)
  img.save(dir_path + './images/' + str(i) + '.png')

f.close()
