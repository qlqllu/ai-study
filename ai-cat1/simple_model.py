import numpy as np

def sigmoid(z):
  return 1/(1 + np.exp(-z))

def train(x, y, iterations = 100, learning_rate = 0.05):
  n = x.shape[0]
  m = x.shape[1]

  w = np.zeros((n, 1))
  b = 0
  cost = 0
  costs = []
  for i in range(iterations):
    z = np.dot(w.T, x) + b
    a = sigmoid(z)
    j = 0
    cost = -1 * (np.sum((y * np.log(a)) + ((1 - y) * (np.log(1 - a))),axis=1))/m
    
    costs.append(cost)
    if i % 100 == 0:
      print ("Cost after iteration %i: %f" %(i, cost))

    dz = a - y
    dw = np.dot(x, dz.T) / m
    db = np.sum(dz) / m

    w = w - learning_rate * dw
    b = b - learning_rate * db

  return w, b, costs

def predict(w, b, x):
  z = np.dot(w.T, x) + b
  a = sigmoid(z)
  p = np.zeros((1, x.shape[1]))
  
  for i in range(a.shape[1]):
    if a[0][i] <= 0.5:
      p[0][i] = 0
    else:
      p[0][i] = 1

  return p

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
  w, b, costs = train(X_train, Y_train, num_iterations, learning_rate)

  predict_train = predict(w, b, X_train)
  predict_test = predict(w, b, X_test)

  print("train accuracy: {} %".format(100 - np.mean(np.abs(predict_train - Y_train)) * 100))
  print("test accuracy: {} %".format(100 - np.mean(np.abs(predict_test - Y_test)) * 100))

  return {
    "w": w,
    "b": b,
    "costs": costs}


