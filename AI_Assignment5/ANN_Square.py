# Course:  CS4242
# Student name:	William Zimmerman
# Student ID: 000731566
# Assignment #: 5a
# Due Date:	04/24/2018
# Signature:______________
# Score:______________


import numpy as numpy
from numpy import dot, random

# Import training data, setup the x_train(expected input) and y_train(expected output)
training_data_input = numpy.array([
    [-1, -1, -1, -1],
    [-1, -1, -1, 1],
    [-1, -1, 1, -1],
    [-1, -1, 1, 1],
    [-1, 1, -1, -1],
    [-1, 1, -1, 1],
    [-1, 1, 1, -1],
    [-1, 1, 1, 1],
    [1, -1, -1, -1],
    [1, -1, -1, 1],
    [1, -1, 1, -1],
    [1, -1, 1, 1],
    [1, 1, -1, -1],
    [1, 1, -1, 1],
    [1, 1, 1, -1],
    [1, 1, 1, 1]])

training_data_output = numpy.array([[0], [0], [0], [1], [0], [1], [1], [1], [0], [1], [1], [1], [1], [1], [1], [1]])

# learning rate
learning_rate = 0.5
# number of iterations(epochs)
learning_iterations = 10000
# number of input from training data
inputlayer_neurons = training_data_input.shape[1]
# number of hidden layers neurons
hiddenlayer_neurons = inputlayer_neurons-1
# number of neurons at output layer
output_neurons = 1


# Sigmoid Function
def sigmoid (x):
    return 1/(1 + numpy.exp(-x))


# Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)


# weight and bias initialization
weight = numpy.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons)) # weights at hidden layer
bias = numpy.random.uniform(size=(1, hiddenlayer_neurons)) # bias at hidden layer
weight_output = numpy.random.uniform(size=(hiddenlayer_neurons, output_neurons)) # weights at output layer
bias_output = numpy.random.uniform(size=(1, output_neurons)) # bias at output layer

print("initial weights: {} initial bias: {} initial output weights: {} initial bias output: {}".format(weight,bias,
                                                                                                       weight_output,
                                                                                                       bias_output))

for i in range(learning_iterations):

    # Forward Propogation
    hidden_layer_output_init = numpy.dot(training_data_input, weight) # calculate hidden layer output using dot product
    hidden_layer_output = hidden_layer_output_init + bias # add bias to above
    hidden_layer_activations = sigmoid(hidden_layer_output) # use the sigmoid fucntion with output
    output_layer_input_init = numpy.dot(hidden_layer_activations, weight_output) # using the above get the hidden layer
    # activation at output layer
    output_layer_input = output_layer_input_init + bias_output # add bias to above
    output = sigmoid(output_layer_input) # use sigmooid funciton at hidden layer at output layer

    # Backpropagation
    result = training_data_output-output # calculate gradient of error at output layer
    slope_output_layer = derivatives_sigmoid(output) # calculate slope at output layer
    slope_hidden_layer = derivatives_sigmoid(hidden_layer_activations) # calculate slop at hidden layer
    back_output = result * slope_output_layer # calulcate delta at output layer
    Error_at_hidden_layer = back_output.dot(weight_output.T) # calucate error at hidden layer
    back_hidden_layer = Error_at_hidden_layer * slope_hidden_layer # calculate delata at hidden layer
    weight_output += hidden_layer_activations.T.dot(back_output) * learning_rate # update weight at output layer
    bias_output += numpy.sum(back_output, axis=0, keepdims=True) * learning_rate # update bias at ou
    weight += training_data_input.T.dot(back_hidden_layer) * learning_rate # update weight at hidden layer
    bias += numpy.sum(back_hidden_layer, axis=0, keepdims=True) * learning_rate # update bias at hidden layer

print("input: {} output: {} squash: {}".format(training_data_input[:16], output, sigmoid(result))) # print out info
print("weights: {} bias: {}".format(weight, bias)) # print out weights and bias at hidden layer
print("expected output: {}".format(training_data_output[:16])) # print out expected output to compare
