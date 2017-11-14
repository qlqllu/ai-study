# Package imports
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from testCases import *
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1) # set a seed so that the results are consistent

X, Y = load_planar_dataset()
print ('The shape of X is: ' + str(X.shape))
print ('The shape of Y is: ' + str(Y.shape))
# plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral)
# plt.show()

def sigmoid(z):
  return 1/(1 + np.exp(-z))

# 4.1
def layer_sizes(X, Y):
  n_x = X.shape[0]
  n_h = 4
  n_y = Y.shape[0]
  return (n_x, n_h, n_y)

X_assess, Y_assess = layer_sizes_test_case()
(n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))

# 4.2
def initialize_parameters(n_x, n_h, n_y):
  parameters = {"W1": np.random.random((n_h, n_x)),
                "b1": np.zeros((n_h, 1)),
                "W2": np.random.random((n_y, n_h)),
                "b2": np.zeros((n_y, 1))}
  return parameters


n_x, n_h, n_y = initialize_parameters_test_case()
parameters = initialize_parameters(n_x, n_h, n_y)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

# 4.3
def forward_propagation(X, parameters):
  Z1 = np.dot(parameters["W1"], X) + parameters["b1"]
  A1 = sigmoid(Z1)
  Z2 = np.dot(parameters["W2"], A1) + parameters["b2"]
  A2 = sigmoid(Z2)
  ### END CODE HERE ###
  
  assert(A2.shape == (1, X.shape[1]))
  
  cache = {"Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2}
  
  return A2, cache

X_assess, parameters = forward_propagation_test_case()
A2, cache = forward_propagation(X_assess, parameters)
# Note: we use the mean here just to make sure that your output matches ours. 
print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))

def compute_cost(A2, Y, parameters):
    m = Y.shape[1] # number of example
    cost = -1 * (np.sum((Y * np.log(A2)) + ((1 - Y) * (np.log(1 - A2))),axis=1))/m
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17
    return cost

A2, Y_assess, parameters = compute_cost_test_case()
print("cost = " + str(compute_cost(A2, Y_assess, parameters)))

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    dZ2= A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis = 1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * (A1 * (1 - A1))
    dW1 = np.dot(X, dZ1.T) / m
    db1 = np.sum(dZ1, axis = 1, keepdims=True) / m
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
             
    return grads

parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
grads = backward_propagation(parameters, cache, X_assess, Y_assess)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dW2 = "+ str(grads["dW2"]))
print ("db2 = "+ str(grads["db2"]))