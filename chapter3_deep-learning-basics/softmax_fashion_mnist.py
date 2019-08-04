from mxnet import nd
import fashion_mnist as fm

batch_size = 256
train_iter, test_iter = fm.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

W = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
b = nd.zeros(num_outputs)

W.attach_grad()
b.attach_grad()

def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp/partition

def net(X):
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)

def cross_entropy(y_hat, y):
    return -nd.pick(y_hat, y).log()

def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()

num_epochs, lr = 10, 0.01
fm.train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

for X, y in test_iter:
    break
true_labels = fm.get_fashion_mnist_labels(y.asnumpy())
pre_labels = fm.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
title = [true + '\n' + pre for true, pre in zip(true_labels, pre_labels)]
fm.show_fashion_mnist(X[:10], title[:10])
