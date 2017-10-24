import numpy as np
import h5py

def sigmoid(z):
  return 1/(1 + np.exp(-z))

f = h5py.File('train_cat.h5', 'r')
train_set_x_orig = np.array(f['train_set_x'])
f.close()

## n: features in one sample
## m: the samples count
b = 0
m = train_set_x_orig.shape[0]
x1 = train_set_x_orig[0]
n = x1.shape[0] * x1.shape[1] * x1.shape[2]

train_set_x = train_set_x_orig.reshape(n, m)

w = np.zeros((n, 1))
z = np.dot(w.T, train_set_x) + b
a = sigmoid(z)

print('train_set_x.shape: ' + str(train_set_x.shape))
print('w.shape: ' + str(w.shape))
print('z.shape: ' + str(z.shape))
print('a.shape: ' + str(a.shape))
print('a: ')
print(a)
