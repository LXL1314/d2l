from mxnet import nd, init
from mxnet.gluon import nn

class MLP(nn.Block):

    def __init__(self, **kwargs):
        super().__init__(**kwargs) #若无这一行：AttributeError: 'MLP' object has no attribute '_children'
        self.hideen = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hideen(x))

net = MLP()
net.initialize()
X = nd.random.normal(scale=1, shape=[3, 5])
print("MLP: ", net(X))
print("net.collect_params()", net.collect_params())
"""
  net.collect_params() mlp0_ (
  Parameter dense0_weight (shape=(256, 5), dtype=float32)
  Parameter dense0_bias (shape=(256,), dtype=float32)
  Parameter dense1_weight (shape=(10, 256), dtype=float32)
  Parameter dense1_bias (shape=(10,), dtype=float32)
)"""
print("weight data1", net.hideen.weight.data())
print("weight data2", net.collect_params()['dense0_weight'].data())

class MySequential(nn.Block):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add(self, *blocks):
        for block in blocks:
            self.register_child(block)

    def forward(self, x):
        for block in self._children.values():
            x = block(x)
        return x

net = MySequential()
net.add(nn.Dense(256, activation='relu'), nn.Dense(10))
#net.add(nn.Dense(10))
net.initialize()
X = nd.random.normal(scale=1, shape=[3, 5])
print("MySequential: ", net(X))
print("net.collect_params()", net.collect_params())
"""
  net.collect_params()
  Parameter dense2_weight (shape=(256, 5), dtype=float32)
  Parameter dense2_bias (shape=(256,), dtype=float32)
  Parameter dense3_weight (shape=(10, 256), dtype=float32)
  Parameter dense3_bias (shape=(10,), dtype=float32)

"""
print("weight data", net.collect_params()['dense2_weight'].data())



class FancyMLP(nn.Block):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rand_weight = self.params.get_constant('rand_weight', nd.random.uniform(shape=(20, 20)))
        #nd.random.uniform() : uniform distribution
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, x):
        x = self.dense(x)
        x = nd.relu(nd.dot(x, self.rand_weight.data()) + 1)
        x = self.dense(x)
        while x.norm().asscalar() > 1:
            x /= 2
        if x.norm().asscalar() < 0.8:
            x *= 10
        return x.sum()

net = nn.Sequential()
net.add(nn.Dense(5, activation='relu'))
net.add(nn.Dense(10))
net.initialize()  # 使用默认初始化方式

X = nd.random.uniform(shape=(2, 20))
Y = net(X)
print("Y：", Y)
print("weight.data", net[0].weight.data())
print("bias.data", net[0].bias.data())

class MyInit(init.Initializer):

    def _init_weight(self, _, data):
        print("init weight")
        data[:] = nd.random.uniform(low=-10, high=10, shape=data.shape)
        data *= data.abs() >= 5

net = nn.Sequential()
net.add(nn.Dense(5, in_units=5, activation='relu'))
net.initialize(MyInit())
print("init 1")
X1 = nd.random.uniform(shape=(2, 5))
Y1 = net(X1)
print("X1: ", X1, "Y1:", Y1)
#print("init 2")
net.initialize(MyInit(), force_reinit=True)
X2 = nd.random.uniform(shape=(10, 5))
Y2 = net(X2)
print("X2: ", X2, "Y2:", Y2)

#print("weight.data", net[0].weight.data()[0])

