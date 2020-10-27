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



