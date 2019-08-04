import function as f
from mxnet import nd, autograd
from mxnet.gluon import nn

X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = nd.array([[0, 1], [2, 3]])
Y = f.corr2d(X, K)
print("Y:", Y)

print("#"*100)

class Conv2D(nn.Block):

    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return f.corr2d(x, self.weight.data()) + self.bias.data()
        #return nd.Convolution(data=x, kernel=self.weight.data(), bias=self.bias.data())

X = nd.ones([6, 8])
X[:, 2:6] = 0

K = nd.array([[1, -1]])
Y = f.corr2d(X, K)
print("Y: ", Y)
#从白到黑的边缘和从黑到白的边缘分别检测成了1和-1。其余部分的输出全是0。

print("#"*100)

conv2d = nn.Conv2D(1, kernel_size=(1, 2))
conv2d.initialize()

"""
conv2d = Conv2D(kernel_size=(1, 2))
conv2d.initialize()
"""

X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        print("Y_hat.shape: ", Y_hat.shape)
        lo = (Y_hat - Y) ** 2
    lo.backward()
    conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()
    if (i + 1) % 2 == 0:
        print('batch %d, loss %.3f' % (i + 1, lo.sum().asscalar()))

print("weight:", conv2d.weight.data().reshape([1, 2]))









