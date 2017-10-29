import numpy as np
import h5py
from PIL import Image
import os
import math
import sys
import matplotlib.pyplot as plt

sys.path.append('.')
import simple_model

def read_train_data():
  f_train = h5py.File('train_cat.h5', 'r')
  train_set_x_orig = np.array(f_train['train_set_x'])
  train_set_y_orig = np.array(f_train['train_set_y'])
  f_train.close()

  ## n: features in one sample
  ## m: the samples count

  m = train_set_x_orig.shape[0]
  x1 = train_set_x_orig[0]
  n = x1.shape[0] * x1.shape[1] * x1.shape[2]

  train_set_x = train_set_x_orig.reshape(m, -1).T
  train_set_y = train_set_y_orig.reshape(1, m)

  return train_set_x/255, train_set_y

def read_test_data():
  f_test = h5py.File('test_cat.h5', 'r')
  test_set_x_orig = np.array(f_test['test_set_x'])
  test_set_y_orig = np.array(f_test['test_set_y'])
  f_test.close()

  m = test_set_x_orig.shape[0]
  x1 = test_set_x_orig[0]
  n = x1.shape[0] * x1.shape[1] * x1.shape[2]

  test_set_x = test_set_x_orig.reshape(m, -1).T
  test_set_y = test_set_y_orig.reshape(1, m)

  return test_set_x/255, test_set_y


test_set_x, test_set_y = read_test_data()
train_set_x, train_set_y = read_train_data()

learning_rates = [0.001, 0.0001]
results = {}
for i in learning_rates:
  print('learning_rate:' + str(i))
  results[str(i)] = simple_model.model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 3000, learning_rate = i)

for i in learning_rates:
    plt.plot(np.squeeze(results[str(i)]["costs"]), label= str(i))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()

dir_path = os.path.dirname(__file__)

def write_train_images(a):
  f = h5py.File('train_cat.h5', 'r')

  for i, data in enumerate(f['train_set_x']):
    img = Image.fromarray(data)
    img.save(dir_path + './images_train/' + str(int(a[0][i])) + '_' + str(i) + '.png')

def write_test_images(a):
  f = h5py.File('test_cat.h5', 'r')

  for i, data in enumerate(f['test_set_x']):
    img = Image.fromarray(data)
    img.save(dir_path + './images_test/' + str(int(a[0][i])) + '_' + str(i) + '.png')

# write_train_images(predict_train)
# write_test_images(predict_test)


