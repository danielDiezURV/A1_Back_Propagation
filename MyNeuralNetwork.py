import numpy as np
from enum import Enum

# Neural Network class
class MyNeuralNetwork:
  def __init__(self, layers, nEpochs, learning_rate, momentum, activation, validation_set):
    # Neural network architecture
    self.L = len(layers)                  # number of layers
    self.n = layers.copy()                # number of neurons in each layer
    self.h = []                           # units field
    self.xi = []                          # units activation
    self.theta = []                       # thresholds
    self.delta = []                       # propagation error
    self.d_theta = []                     # thresholds delta
    self.w = []                           # edge weights
    self.d_w = []                         # weights delta
    self.d_w_prev = []                    # previous weights delta
    # Learning params
    self.nEpochs = nEpochs                # number of epochs
    self.learning_rate = learning_rate    # learning rate
    self.momentum = momentum              # momentum
    self.activation = activation          # activation function (sigmoid, relu, linear, tanh)
    self.validation_set = validation_set  # Percentage of data used for validation
  
    for l in range(self.L):
      self.h.append(np.zeros(layers[l]))
      self.xi.append(np.zeros(layers[l]))
      self.theta.append(np.random.rand(layers[l]))  # random, but should have also negative values
      self.delta.append(np.zeros(layers[l]))
      self.d_theta.append(np.zeros(layers[l]))
  
    for l in range(1, self.L):
      self.w.append(np.random.rand(layers[l], layers[l - 1]))  # random, but should have also negative values
      self.d_w.append(np.zeros((layers[l], layers[l - 1])))
      self.d_w_prev.append(np.zeros((layers[l], layers[l - 1])))

    self.d_theta_prev = self.d_theta

  #TODO -> create activation funtions

  #TODO -> create learning functions

  #TODO -> create predict function

  # layers include input layer + hidden layers + output layer
layers = [4, 9, 5, 1]
nn = MyNeuralNetwork(layers)

print("L = ", nn.L, end="\n")
print("n = ", nn.n, end="\n")

print("xi = ", nn.xi, end="\n")
print("xi[0] = ", nn.xi[0], end="\n")
print("xi[1] = ", nn.xi[0], end="\n")

print("wh = ", nn.w, end="\n")
print("wh[1] = ", nn.w[1], end="\n")