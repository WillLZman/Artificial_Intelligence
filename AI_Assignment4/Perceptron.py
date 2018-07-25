# Course:  CS4242
# Student name:	William Zimmerman
# Student ID: 000731566
# Assignment #: 4
# Due Date:	04/17/2018
# Signature:______________
# Score:______________
from random import choice
from numpy.core.tests.test_mem_overlap import xrange
from numpy import dot, random, array

# Input arrays arranged in a tuple, first element being the input array last element being the expected output
# first 4 elements are inputs, last element is expected value, 0 = dark expectation 1 = light expectation
# -1 = dark square 1 = bright square
training_data = [
    (array([-1, -1, -1, -1]), 0),
    (array([-1, -1, -1, 1]), 0),
    (array([-1, -1, 1, -1]), 0),
    (array([-1, -1, 1, 1]), 1),
    (array([-1, 1, -1, -1]), 0),
    (array([-1, 1, -1, 1]), 1),
    (array([-1, 1, 1, -1]), 1),
    (array([-1, 1, 1, 1]), 1),
    (array([1, -1, -1, -1]), 0),
    (array([1, -1, -1, 1]), 1),
    (array([1, -1, 1, -1]), 1),
    (array([1, -1, 1, 1]), 1),
    (array([1, 1, -1, -1]), 1),
    (array([1, 1, -1, 1]), 1),
    (array([1, 1, 1, -1]), 1),
    (array([1, 1, 1, 1]), 1),
]

# set the weight of the 4 input neurons to activation function
weight = random.rand(4)
# learning rate for the perceptron
learning_rate = 0.01
# number of iterations(epochs) for the percepetron,
learning_iterations = 100


# step function that will determine the output
def activation_function(x):
    if x < -0.1:
        return 0
    else:
        return 1


# The below nested for loop(using xrange as to my knowledge it's slightly faster) runs the training data the number of
# times specified by the learning iterations above So based of the training data it identifies the input and expected
# output based off the tuples in the training data Then it computes the result of the output using the dot product as
# we are using an array for representing the square Then it calculates the error by taking the expected value from the
# training data and subtracts the result after squashing it in the activation(step) function from above. Then it uses
# the error and adjusts the weight


for i in xrange(learning_iterations):
    x, expected = choice(training_data)
    result = dot(weight, x)
    error = expected - activation_function(result)
    weight += learning_rate * error * x
# print out weights after learning from training data
print("Learning Rate: ", learning_rate, "Learning_Iterations: ", learning_iterations)
print("weights:", weight)
# The below for loop simply iterates through the training data and prints out the input, results, and squashed result
for x, _ in training_data:
    result = dot(x, weight)
    print("input: {} output: {} squash: {}".format(x[:4], result, activation_function(result)))
