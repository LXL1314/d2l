from mxnet import nd
from mxnet.gluon import nn


def comp_conv2d(conv2d, X):
    conv2d.initialize()
    ##对输入进行升维，（1， 1）代表的是批量大小和通道数
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    ##对Y进行降维，
    return Y.reshape(shape=(Y.shape[2:]))

conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
X = nd.random.uniform(shape=(8, 8))
Y = comp_conv2d(conv2d, X)
print("Y:", Y.shape)

conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
Y = comp_conv2d(conv2d, X)
print("Y:", Y.shape)



