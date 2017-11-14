import numpy as np

def sigmoid(z):
  return 1/(1 + np.exp(-z))

def layer_sizes(X, Y):
  n_x = X.shape[0]
  n_y = Y.shape[0]
  return (n_x, n_y)

def initialize_parameters(n_x, n_h, n_y):
  parameters = {"W1": np.random.random((n_h, n_x)) * 0.01,
                "b1": np.zeros((n_h, 1)),
                "W2": np.random.random((n_y, n_h)) * 0.01,
                "b2": np.zeros((n_y, 1))}
  return parameters

def forward_propagation(X, parameters):
  Z1 = np.dot(parameters["W1"], X) + parameters["b1"]
  A1 = sigmoid(Z1)
  Z2 = np.dot(parameters["W2"], A1) + parameters["b2"]
  A2 = sigmoid(Z2)
  
  cache = {"Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2}
  
  return A2, cache

def compute_cost(A2, Y, parameters):
  m = Y.shape[1] # number of example
  cost = -1 * (np.sum((Y * np.log(A2)) + ((1 - Y) * (np.log(1 - A2))),axis=1))/m
  cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                              # E.g., turns [[17]] into 17
  return cost

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    dZ2= A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis = 1, keepdims=True) / m
    dZ1 = np.multiply(np.dot(W2.T, dZ2), (A1 * (1 - A1)))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis = 1, keepdims=True) / m
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
             
    return grads

def update_parameters(parameters, grads, learning_rate):
  new_parameters = {"W1": parameters["W1"] - learning_rate * grads["dW1"],
                    "b1": parameters["b1"] - learning_rate * grads["db1"],
                    "W2": parameters["W2"] - learning_rate * grads["dW2"],
                    "b2": parameters["b2"] - learning_rate * grads["db2"]}
  return new_parameters

def predict(X, parameters):
  A2, cache = forward_propagation(X, parameters)

  return np.around(A2)

def train(X, Y, n_h, num_iterations = 10000, learning_rate = 1.2, print_cost=False):
  np.random.seed(3)
  n_x = layer_sizes(X, Y)[0]
  n_y = layer_sizes(X, Y)[1]
  
  parameters = initialize_parameters(n_x, n_h, n_y)
  
  costs = []
  for i in range(0, num_iterations):
    
    A2, cache = forward_propagation(X, parameters)
    
    cost = compute_cost(A2, Y, parameters)
    costs.append(cost)
    grads = backward_propagation(parameters, cache, X, Y)

    parameters = update_parameters(parameters, grads, learning_rate)
    
    # Print the cost every 1000 iterations
    if print_cost and i % 100 == 0:
      print ("Cost after iteration %i: %f" %(i, cost))
      
  return parameters, costs

def model(X_train, Y_train, n_h, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = True):
  parameters, costs = train(X_train, Y_train, n_h, num_iterations, learning_rate, print_cost)

  predict_train = predict(X_train, parameters)
  predict_test = predict(X_test, parameters)

  print("train accuracy: {} %".format(100 - np.mean(np.abs(predict_train - Y_train)) * 100))
  print("test accuracy: {} %".format(100 - np.mean(np.abs(predict_test - Y_test)) * 100))

  return {
    "parameters": parameters,
    "costs": costs}