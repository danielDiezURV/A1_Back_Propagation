from enum import Enum
import numpy as np

# Activation functions
class ActivationFunction(Enum):
  RELU = "RELU",lambda x: np.maximum(0, x)
  TANH = "TANH",lambda x: np.tanh(x)
  SIGMOID = "SIGMOID",lambda x: 1 / (1 + np.exp(-x))
  LINEAR = "LINEAR",lambda x: x

  @classmethod
  def get_function(cls, name):
    for function in cls:
      if function.value[0] == name:
        return function.value[1]
    raise ValueError(f"{name} is not a valid activation function")
  
# Neural Network class
class MyNeuralNetwork:
  def __init__(self, layers, nEpochs, learning_rate, momentum, activation, validation_set):
    # Neural network architecture
    self.L = len(layers)                                            # number of layers
    self.n = layers.copy()                                          # number of neurons in each layer
    self.h = []                                                     # units field
    self.xi = []                                                    # units activation
    self.w = []                                                     # edge weights
    self.theta = []                                                 # thresholds
    self.delta = []                                                 # propagation error
    self.d_w = []                                                   # weights delta
    self.d_theta = []                                               # thresholds delta
    self.d_w_prev = []                                              # previous weights delta
    # Learning params
    self.nEpochs = nEpochs                                          # number of epochs
    self.learning_rate = learning_rate                              # learning rate
    self.momentum = momentum                                        # momentum
    self.activation = ActivationFunction.get_function(activation.upper())  # activation function (sigmoid, relu, linear, tanh)
    self.validation_set = validation_set                            # Percentage of data used for validation

    # Initialize Network
    self.xi.append(np.zeros(layers[0]))                             # input layer
    for l in range(self.L):
      self.h.append(np.zeros(layers[l]))
      self.xi.append(np.zeros(layers[l]))
      self.theta.append(np.random.rand(layers[l])) 
      self.delta.append(np.zeros(layers[l]))
      self.d_theta.append(np.zeros(layers[l]))

    # Initialize weights
    self.w.append(np.random.rand(layers[0], layers[0])) 
    self.d_w.append(np.zeros((layers[0], layers[0])))
    self.d_w_prev.append(np.zeros((layers[0], layers[0])))
    for l in range(1, self.L):
      self.w.append(np.random.rand(layers[l], layers[l - 1])) 
      self.d_w.append(np.zeros((layers[l], layers[l - 1])))
      self.d_w_prev.append(np.zeros((layers[l], layers[l - 1])))

    self.d_theta_prev = self.d_theta

  def fit(self, x, y):
    loss = []
    for epoch in range(self.nEpochs):
      for i in np.random.permutation(len(x)):                      # for each random pattern xµ in the training set
        x_prediction = self.feedForward(x.values[i])                     # feed-forward the network with xµ
        
        # TODO: ↓↓↓↓↓
        # evaluate the network output yµ
        # Back-propagate the error for this pattern
        # Update the weights and thresholds        
      # Feed-forward all training patterns and calculate their prediction quadratic error
      # Feed-forward all validation patterns and calculate their prediction quadratic error
    # Optional: Plot the evolution of the training and validation errors  
    # Feed-forward all test patterns
    # Descale the predictions of test patterns, and evaluate them

  def feedForward(self, x):
    # Copy input values to input layer
    self.xi[0] = x
    # Feed-forward first layer
    for i in range(self.n[0]):
        self.h[0][i] = sum(self.w[0][i][j] * self.xi[0][j] for j in range(self.n[0])) - self.theta[0][i]
        self.xi[1][i] = self.activation(self.h[0][i])
    # Feed-forward hidden layers
    for l in range(1, self.L):
      for i in range(self.n[l]):
        self.h[l][i] = sum(self.w[l][i][j] * self.xi[l][j] for j in range(self.n[l-1])) - self.theta[l][i]
        self.xi[l+1][i] = self.activation(self.h[l][i])
    # Return output layer
    return self.xi[self.L]
    


  
