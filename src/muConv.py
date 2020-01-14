# Libraries
# Standard library
from bs4 import BeautifulSoup
import numpy as np
import requests
import PIL.Image
import urllib
import cv2
import random
import matplotlib.pyplot as plt
from urllib.error import HTTPError
from tempfile import TemporaryFile
from scipy import signal
import math


def url_to_image(url, dim):
    try:
        resp = urllib.request.urlopen(url)
    except HTTPError as err:
        print("Image {} is not loaded. Error {}".format(url, err.code))
    else:
        img = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
        return img


def loadImageNetBuilding():
    page = requests.get(
        "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n14785065")  # ship synset
    print(page.content)
    # puts the content of the website into the soup variable, each url on a different line
    soup = BeautifulSoup(page.content, 'html.parser')
    str_soup = str(soup)  # convert soup to string so it can be split
    urls = str_soup.splitlines()
    urls = urls[:-1]
    img_rows, img_cols = 96, 96
    input_shape = (img_rows, img_cols)

    img_matrixes = tuple(url_to_image(urls[idx], input_shape)
                         for idx in range(3))  # range(len(urls)))

    np.savez("outfile.npz", img_matrixes)
    data = np.load("outfile.npz", allow_pickle=True)
    for idxFiles in data.files:
        arr = data[idxFiles]
        print(arr)


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
            # random.shuffle(training_data)

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


class ConvLayer(object):
    def __init__(self, inputLayer, filters, biases):
        self.nextLayerDim = Helper.get_nextLayerDim(
            inputLayer.shape[0], filters[0].shape[0], 0, 1)

        self.filters = filters
        self.inputLayer = inputLayer
        self.biases = biases

    def run(self):
        output_collection = [signal.convolve(self.inputLayer, filter, mode='valid', method='auto')
                             for filter in zip(self.filters, self.biases)]
        self.output = np.stack(output_collection)
        return self.output


class ActivationLayer(object):
    def __init__(self, inputLayer, activationLogic):
        self.inputLayer = inputLayer
        self.activationLogic = activationLogic

    def run(self):
        self.output = self.activationLogic.function(self.inputLayer)
        return self.output


class PoolLayer(object):
    def __init__(self, inputLayer, poolingLogic):
        self.inputLayer = inputLayer
        self.poolingLogic = poolingLogic

    def run(self):
        self.output = self.poolingLogic(self.inputLayer)
        return self.output

def MaxPooling(z):
    output_dim = z.shape[0]/2 if z.shape[0]%2==0 else (z.shape[0]+1)/2
    output = np.zeros((output_dim, output_dim, z.shape[2]))
    for d in range(z.shape[2]):
        for i in range(z.shape[0]):
            for j in range(z.shape[0]):
                output[i,j,d] = z[i:i+2, j:j+2, d].max()
    return output


class ActivationLogicReLU(object):
    @staticmethod
    def function(z):
        return max(0, z)

    @staticmethod
    def function_prime(z):
        return 1 if z > 0 else 0


class ActivationLogicSoftplus(object):
    @staticmethod
    def function(z):
        return math.log(1+math.exp(z))

    @staticmethod
    def function_prime(z):
        return 1/(1+math.exp(-z))


class Helper(object):
    @staticmethod
    def get_nextLayerDim(inputDim, filterDim, padding, step):
        return (inputDim-filterDim+2*padding)/step+1


loadImageNetBuilding()
