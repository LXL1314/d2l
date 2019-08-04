"""

缓解卷积层对位置的过度敏感性

"""

from mxnet import nd
from mxnet.gluon import nn

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = nd.zeros(shape=(X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
Y = pool2d(X, (2, 2), 'avg')
print("Y:", Y)

X = nd.arange(16).reshape(shape=(1, 1, 4, 4))
pool2d = nn.MaxPool2D(3)#池化窗口为（3， 3）， 步幅形状默认为（3， 3)
Y = pool2d(X)
print("Y:", Y)

pool2d = nn.MaxPool2D(3, padding=1, strides=2)#池化窗口为（3， 3）, 宽和高两边各填充1， 高和宽的步幅为2
Y = pool2d(X)
print("Y:", Y)

pool2d = nn.MaxPool2D((2, 3), padding=(1, 2), strides=(2, 3))
Y = pool2d(X)
print("Y:", Y)

























