import matplotlib.pyplot as plt
import requests, gzip, os, hashlib
import numpy as np


def fetch_dataset(url):
    import requests, gzip, os, hashlib, numpy
    fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            dat = f.read()
    else:
        with open(fp, "wb") as f:
            dat = requests.get(url).content
            f.write(dat)
    return numpy.frombuffer(gzip.decompress(dat), dtype=numpy.uint8).copy()
#return np.frombuffer(gzip.decompress(data), dtype=np.uint8)




def get_mnist():
    X_train = fetch_dataset("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28,28))
    Y_train = fetch_dataset("http://yann.lecun.com/exdb/mnist/train-images-idx1-ubyte.gz")[8:]
    X_test = fetch_dataset("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28,28))
    Y_test = fetch_dataset("http://yann.lecun.com/exdb/mnist/t10-images-idx1-ubyte.gz")[8:]
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = get_mnist()


def connected_layer(inputs, outputs):
    return np.ones([inputs, outputs])

def neural_net():
    ''' creates simple neural net '''
    first_layer = connected_layer(2, 3)
    second_layer = connected_layer(3, 3)
    output_layer = connected_layer(3, 2)
    return [first_layer, second_layer, output_layer]

os.getcwd()

import tensor.neural_net  

print(neural_net())

neural_net()


def reLU(layer):
    return np.maximum(np.zeros([layer.shape[0],layer.shape[1]]),layer)

def soft_max(input):
    return np.exp(input) / np.sum(np.exp(input), axis=0)

def generate_random_dataset(n):
    x = np.ones([n,1])
    y = np.random.randint(2,size=n)
    return np.array(y,x-y)


def forward_pass(input, neural_net):
    activation = input
    for weight_layer in neural_net:
        weighted_sum = np.dot(weight_layer.T, activation)
        activation = reLU(weighted_sum)
    return soft_max(activation)

pred = (forward_pass(np.ones([2,1]), neural_net()))


from math import log2

def cross_entropy(pred, label):
    return -sum([label[i]*log2(pred[i]) for i in range(len(pred))])

print(cross_entropy(pred, np.array([0,3])))


# backprop



