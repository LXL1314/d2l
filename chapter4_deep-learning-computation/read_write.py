from mxnet import nd
from mxnet.gluon import nn

x = nd.ones(3)
nd.save('x', x)
print("x: ", x)
x2, = nd.load('x')
print("x2: ", x2)

y = nd.zeros(4)
nd.save('xy', [x, y])
print("xy: ", nd.load('xy'))

x1, y1 = nd.load('xy')
print("x1: ", x1)
print("y1: ", y1)

mydict = {'x': x, 'y': y}
print("mydict:", mydict)
nd.save("mydict", mydict)
mydict2 = nd.load("mydict")
print("mydict2:", mydict2)

class MLP(nn.Block):

    def __init__(self, **kwargs):
        super().__init__(**kwargs) #若无这一行：AttributeError: 'MLP' object has no attribute '_children'
        self.hideen = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hideen(x))

net = MLP()
net.initialize()
X = nd.random.normal(scale=2, shape=(3, 6))
Y = net(X)
print("perams: ", net.collect_params())
net.save_parameters("mlp.parameters")
##save_params() 已经被弃用，要保存参数用save_parameters()
params = net.load_parameters("mlp.parameters")
print("params", params)##params None
net2 = MLP()
net2.load_parameters("mlp.parameters")
print(Y == net2(X))
