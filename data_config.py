import random
from mxnet import nd
def data_iter(batch_size, data, labels):
    num_examples = len(data)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i:min(i + batch_size, num_examples)])
        yield data.take(j), labels.take(j)

def linearRegression(X, w, b):
    return nd.dot(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
    #y_hat shape: [m, 1]
    #y shape: [m, ]
    #(y_hat - y ) shape: 广播机制：[m, m]

def SGD(params, lr, batch_size):
    for param in params:
        param[:] = param - (lr / batch_size) * param.grad

