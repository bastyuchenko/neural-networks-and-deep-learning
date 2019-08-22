# Libraries
# Standard library
import random
import matplotlib.pyplot as plt

# Third-party libraries
import numpy as np


class Network(object):
    def __init__(self, sizes):
        self.sizes = sizes
        np.random.seed(123)
        self.weights = [np.random.rand(i, o)
                        for i, o in zip(sizes[1:], sizes[:-1])]

        self.biases = [np.random.rand(d, 1)
                       for d in sizes[1:]]

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        cost = []
        for ep in range(epochs):
            print("Epoch {} complete".format(ep))
            np.random.seed(123)
            #random.shuffle(training_data)

            for k in range(0, len(training_data), mini_batch_size):
                mini_batch = training_data[k:k+mini_batch_size]
                self.update_mini_batch(mini_batch, eta)
                print("Mini-Batch {} complete".format(k))

            cost.append(sum(np.array(calculate_cost(
                training_data, self.weights, self.biases)).flatten()))
        plt.plot(cost)
        plt.show()

    def update_mini_batch(self, mini_batch, eta):
        nuble_Ws = [np.zeros(w.shape) for w in self.weights]
        nuble_Bs = [np.zeros(b.shape) for b in self.biases]

        for x, y in mini_batch:
            bnws, bnbs = self.backprop(x, y)
            nuble_Ws = [nws+bnw for nws, bnw in zip(nuble_Ws, bnws)]
            nuble_Bs = [nbs+bbw for nbs, bbw in zip(nuble_Bs, bnbs)]

        self.weights = [w-(eta/len(mini_batch)*nw)
                        for w, nw in zip(self.weights, nuble_Ws)]
        self.biases = [b-(eta/len(mini_batch)*nb)
                       for b, nb in zip(self.biases, nuble_Bs)]

    def backprop(self, x, y):
        derivative_by_w = [np.zeros_like(w) for w in self.weights]
        derivative_by_b = [np.zeros_like(b) for b in self.biases]
        zz = []
        aa = [x]

        for w, b in zip(self.weights, self.biases):
            z = w@aa[-1] + b
            a = sigmoid(z)
            zz.append(z)
            aa.append(a)

        d = (aa[-1]-y)*sigmoid_prime(zz[-1])
        derivative_by_b[-1] += d
        derivative_by_w[-1] += d@aa[-2].T

        for l in reversed(range(1, len(self.sizes)-1)):
            d = self.weights[l].T@d*sigmoid_prime(zz[l-1])
            derivative_by_b[l-1] += d
            derivative_by_w[l-1] += d@aa[l-1].T

        return (derivative_by_w, derivative_by_b)


def calculate_cost(training, w, b):
    return sum([calculate_output(y, x, w, b) for x, y in training])/len(training)


def calculate_output(y, x, w, b):
    a = x
    for wl, bl in zip(w, b):
        a = sigmoid(wl@a+bl)
    return (y-a)**2


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
