from mxnet import gluon, nd
from mxnet.gluon import nn

class MyDense(nn.Block):

    def __init__(self, units, in_units, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        x = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(x)

net = MyDense(units=4, in_units=6)
net.initialize()
X = nd.random.normal(shape=[3, 6])
Y = net.forward(X)
print(Y)


net1 = nn.Sequential()
net1.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net1.initialize()
X = nd.random.uniform(shape=(2, 64))
Y = net1(X)
print(net1[0].weight.shape)
print(Y)
net2 = nn.Sequential()
net2.add(nn.Dense(8, in_units=64),
        MyDense(1, in_units=8))
net2.initialize()
Y = net2(X)
print(net2[0].weight.shape)
print(net2[1].weight.shape)
print(Y)
