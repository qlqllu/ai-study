import numpy as np
import h5py

def sigmoid(z):
  return 1/(1 + np.exp(-z))

f = h5py.File('train_cat.h5', 'r')
train_set_x_orig = np.array(f['train_set_x'])
train_set_y_orig = np.array(f['train_set_y'])
f.close()

## n: features in one sample
## m: the samples count

m = train_set_x_orig.shape[0]
x1 = train_set_x_orig[0]
n = x1.shape[0] * x1.shape[1] * x1.shape[2]

train_set_x = train_set_x_orig.reshape(m, -1).T
train_set_y = train_set_y_orig.reshape(1, m)

alpha = 0.01
b = 0
w = np.zeros((n, 1))

for i in range(10):
  z = np.dot(w.T, train_set_x) + b
  a = sigmoid(z)
  dz = a - train_set_y
  dw = 1 / m * np.dot(train_set_x, dz.T)
  db = 1 / m * np.sum(dz)

  w -= alpha * dw
  b -= alpha * db

print('w:' + str(w.T))
print('b:' + str(b))

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

