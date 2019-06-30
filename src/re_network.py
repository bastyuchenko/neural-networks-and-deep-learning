"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

# Libraries
# Standard library
import random

# Third-party libraries
import numpy as np


class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        random.seed(123)
        np.random.seed(123)
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        #random.shuffle(training_data)
        self.update_mini_batch(training_data, 0.1)
        return

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*n for w, n in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*n for b, n in zip(self.biases, nabla_b)]
        for w, n in zip(self.weights, nabla_w):
            w = w-(eta/len(mini_batch))*n
        for b, n in zip(self.biases, nabla_b):
            b = b-(eta/len(mini_batch))*n

    def backprop(self, x, y):
        a = x
        activations = [a]
        zs = []
        # feedforward
        for w, b in zip(self.weights, self.biases):
            z = (w@a)+b
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)

        # backward pass
        delta = (activations[-1]-y)*sigmoid_prime(zs[-1])
        nabla_b=[delta]
        nabla_w=[delta@activations[-2].T]
        for l in range(1, len(self.weights)):
            delta = self.weights[-l].T@delta*sigmoid_prime(zs[-l-1])
            nabla_b.insert(0, delta)
            nabla_w.insert(0, delta@activations[-l-2].T)
        return (nabla_b, nabla_w)


# Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
