import numpy as np
import h5py
from PIL import Image
import os

def sigmoid(z):
  return 1/(1 + np.exp(-z))

def train(x, y):
  alpha = 0.01
  iterations = 10
  b = 0
  n = x.shape[0]
  m = x.shape[1]
  w = np.zeros((n, 1))

  for i in range(iterations):
    z = np.dot(w.T, x) + b
    a = sigmoid(z)
    dz = a - y
    dw = 1 / m * np.dot(x, dz.T)
    db = 1 / m * np.sum(dz)

    w -= alpha * dw
    b -= alpha * db

  return w, b

def predict(w, b, x):
  z = np.dot(w.T, x) + b
  a = sigmoid(z)

  return a

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

  return train_set_x, train_set_y

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

  return test_set_x, test_set_y


test_set_x, test_set_y = read_test_data()
train_set_x, train_set_y = read_train_data()

w, b = train(train_set_x, train_set_y)

predict_train = predict(w, b, train_set_x)
predict_test = predict(w, b, test_set_x)

print("train accuracy: {} %".format(100 - np.mean(np.abs(predict_train - train_set_y)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(predict_test - test_set_y)) * 100))


dir_path = os.path.dirname(__file__)

def write_train_images(a):
  f = h5py.File('train_cat.h5', 'r')

  for i, data in enumerate(f['train_set_x']):
    img = Image.fromarray(data)
    img.save(dir_path + './images_train/' + str(i) + '_' + str(a[0][i]) + '.png')

# write_train_images(predict_train)
print(predict_train)
print('................')
print(train_set_y)

# print('train_set_x: ' + str(train_set_x))
# print('train_set_x_orig.shape: ' + str(train_set_x_orig.shape))
# print('train_set_x.shape: ' + str(train_set_x.shape))
# print('train_set_y: ' + str(train_set_y))
# print('train_set_y.shape: ' + str(train_set_y.shape))
# print('w.shape: ' + str(w.shape))
# print('z.shape: ' + str(z.shape))
# print('a.shape: ' + str(a.shape))
# print('dz.shape: ' + str(dz.shape))
# print('dw.shape: ' + str(dw.shape))
# print('db.shape: ' + str(db.shape))

