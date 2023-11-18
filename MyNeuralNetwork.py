from enum import Enum
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

# Activation functions
class ActivationFunction(Enum):
  RELU = "RELU",lambda x: np.maximum(0, x), lambda x: 1 if x >= 0 else 0
  TANH = "TANH",lambda x: np.tanh(x), lambda x: 1 - np.tanh(x) ** 2
  SIGMOID = "SIGMOID",lambda x: 1 / (1 + np.exp(-x)), lambda x: np.exp(-(1 / (1 + np.exp(-x)))) / (1 + np.exp(-(1 / (1 + np.exp(-x)))))
  LINEAR = "LINEAR",lambda x: x, lambda x: 1

  @classmethod
  def get_function(cls, name):
    for function in cls:
      if function.value[0] == name:
        return function.value[1]
    raise ValueError(f"{name} is not a valid activation function")
  
  @classmethod
  def get_derivative(cls, name):
    for function in cls:
      if function.value[0] == name:
        return function.value[2]
    raise ValueError(f"{name} is not a valid activation function")
  
# Neural Network class
class MyNeuralNetwork:
  def __init__(self, layers, nEpochs, learning_rate, momentum, activation, validation_set):
    # Neural network architecture
    self.L = len(layers)                                            # number of layers
    self.n = layers                                                 # number of neurons in each layer
    self.h = []                                                     # units field
    self.xi = []                                                    # units activation
    self.w = []                                                     # edge weights
    self.theta = []                                                 # thresholds
    self.delta = []                                                 # propagation error
    self.w_prev = []                                                # previous weights delta
    self.theta_prev = []                                            # previous thresholds delta
    # Learning params
    self.nEpochs = nEpochs                                          # number of epochs
    self.learning_rate = learning_rate                              # learning rate
    self.momentum = momentum                                        # momentum
    self.activation = ActivationFunction.get_function(activation.upper())  # activation function (sigmoid, relu, linear, tanh)
    self.derivative = ActivationFunction.get_derivative(activation.upper()) # derivative of activation function
    self.validation_set = validation_set                            # Percentage of data used for validation
    # Performance metrics
    self.train_loss = np.zeros(nEpochs)          # train loss
    self.val_loss = np.zeros(nEpochs)                # validation loss

    # Initialize Network
    self.xi.append(np.zeros(layers[0]))                             # input layer
    for l in range(self.L):
      self.h.append(np.zeros(layers[l]))
      self.xi.append(np.zeros(layers[l]))
      self.theta.append(np.random.rand(layers[l])) 
      self.delta.append(np.zeros(layers[l]))


    # Initialize weights
    self.w.append(np.random.rand(layers[0], layers[0])) 
    for l in range(1, self.L):
      self.w.append(np.random.rand(layers[l], layers[l - 1])) 

    self.w_prev = self.w.copy()
    self.theta_prev = self.theta.copy()

  def fit(self, x, y):
    x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size=self.validation_set, shuffle=True)
    for epoch in range(self.nEpochs):
      for i in np.random.permutation(len(x_train)):                      # for each random pattern xµ in the training set
        x_prediction = self.feedForward(x_train[i])               # feed-forward the network with xµ
        self.back_propagation(x_prediction, y_train[i])       # back-propagate the error for this pattern
        self.update_weights()                                      # update the weights and thresholds

      # Feed-forward all training patterns and calculate their prediction quadratic error
      
      (self.train_loss[epoch], train_mae) = self.calculate_error(x_train, y_train)
      (self.val_loss[epoch], val_mae) = self.calculate_error(x_validate, y_validate)

      print("Epoch ",epoch+1,"/",self.nEpochs)
      print("============= [loss: ", round(self.train_loss[epoch], 4), " - mae: ", round(train_mae, 4), " - val_loss: ", round(self.val_loss[epoch], 4), " - val_mae:  ", round(val_mae, 4),"]")

  def predict(self, x):
    predictions = np.zeros((len(x), 1))
    for i in range(len(x)):
      predictions[i][0] = self.feedForward(x[i])
    return predictions

  def loss_epochs(self):  
    return self.train_loss, self.val_loss
  
  def feedForward(self, x):
    # Copy input values to input layer
    self.xi[0] = x
    # Feed-forward hidden layers
    for l in range(self.L):
      for i in range(self.n[l]):
        self.h[l][i] = np.dot(self.w[l][i], self.xi[l]) - self.theta[l][i]
        self.xi[l+1][i] = self.activation(self.h[l][i])
    # Return o utput layer
    return self.xi[self.L]  


  def back_propagation(self, x_predicted, y_expected):
    # Compute output layer error
    for i in range(self.n[self.L-1]):
      self.delta[self.L-1][i] = self.derivative((self.h[self.L-1][i])) * (x_predicted - y_expected)
    # Compute hidden layers error
    for l in reversed(range(1,self.L)):
      for j in range(self.n[l-1]):
        sum_delta_x_weight = 0
        for i in range(self.n[l]):   
          sum_delta_x_weight += self.w[l][i][j] * self.delta[l][i]    
        self.delta[l-1][j] = sum_delta_x_weight * self.derivative(self.h[l-1][j])


  def update_weights(self):
    for i in range(self.n[0]):
      for j in range(self.n[0]):
        self.w_prev[0][i][j] = (self.momentum * self.w_prev[0][i][j]) -(self.learning_rate * self.delta[0][i] * self.xi[1][i]) 
        self.w[0][i][j] += self.w_prev[0][i][j]

    self.theta_prev[0] = (self.momentum * self.theta_prev[0]) - (self.learning_rate * self.delta[0][i])
    self.theta[0] += self.theta_prev[0]

      
    for l in range(1,self.L):
      for i in range(self.n[l]):
        for j in range(self.n[l-1]):
          self.w_prev[l][i][j] = -(self.learning_rate * self.delta[l][i] * self.xi[l+1][i]) + (self.momentum * self.w_prev[l][i][j])
          self.w[l][i][j] += self.w_prev[l][i][j]

      self.theta_prev[l] = -(self.learning_rate * self.delta[l][i]) + (self.momentum * self.theta_prev[l])
      self.theta[l] += self.theta_prev[l]

  def calculate_error(self, x, y):
      predictions = np.zeros(len(x))
      for i in range(len(x)):
        predictions[i] = self.feedForward(x[i])
      return(mean_squared_error(predictions, y), mean_absolute_error(predictions, y))





