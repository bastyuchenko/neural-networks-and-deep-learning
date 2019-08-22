import random
import numpy as np
import sys
import mnist_loader as mnist_loader
import network as network
import re_network as rnetwork

np.random.seed(123)
training_dataX=[np.random.rand(2, 1) for _ in range(5000)]
training_dataY=[np.random.rand(3, 1) for _ in range(5000)]

net = network.Network([2,5,7,3])
rnet = rnetwork.Network([2,5,7,3])
rnet.weights = net.weights.copy()
rnet.biases = net.biases.copy()

net.SGD(list(zip(training_dataX, training_dataY)), 5, 100, 0.1)
print(net.weights)
print(net.biases)
print('*********')

np.random.seed(123)
training_dataX=[np.random.rand(2, 1) for _ in range(5000)]
training_dataY=[np.random.rand(3, 1) for _ in range(5000)]


rnet.SGD(list(zip(training_dataX, training_dataY)), 5, 100, 0.1)
print(rnet.weights)
print(rnet.biases)
print('*********')

'''import random
import numpy as np
import sys
import mnist_loader as mnist_loader
import network as network

full_td, _, _ = mnist_loader.load_data_wrapper()
training_data = [(x[:3], y[:2]) for x, y in full_td] # Just use the first 1000 items of training data
print(training_data[:1])

net = network.Network([784, 30, 30, 10])
net.SGD(training_data[:10], 1, 1000, 0.1)'''