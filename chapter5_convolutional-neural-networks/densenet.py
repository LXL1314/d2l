from mxnet import nd, init
from mxnet.gluon import Trainer, nn
from charpter5 import *

##批量归一化，激活函数，卷积

##稠密块：由一个批量归一化，激活函数，卷积组成
def conv_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'), nn.Conv2D(num_channels, kernel_size=3, padding=1))##大小不变
    return blk

class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))##每层的输出通道不变，但是该层的下一层的输入通道会增加num_channels个通道

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = nd.concat(X, Y, dim=1)
            ##densenet中这里是每块的输出和输入在通道维上连结(所以每个稠密块都会带来通道数的增加, 每一个稠密块增加num_channels个通道数)，
            ##而resnet是每块的输出和输入相加
        return X


##过渡层
def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1),##大小不变，仅用于改变通道数
            nn.AvgPool2D(pool_size=2, strides=2))##高和宽减半
    return blk

##Densenet模型
net = nn.Sequential()
##这一层和resnet一样
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))

##四个稠密块，每个稠密块使用四个卷积层，同resnet一致
num_convs =4
num_channels, growth_rate = 64, 32
for i in range(num_convs):
    net.add(DenseBlock(num_convs, growth_rate))
    num_channels += num_convs * growth_rate
    if i != num_convs-1:
        num_channels //= 2
        net.add(transition_block(num_channels))

net.add(nn.BatchNorm(), nn.Activation('relu'), nn.GlobalAvgPool2D(),
        nn.Dense(10))
net.initialize(init=init.Xavier())


lr, batch_size, num_epochs = 0.1, 256, 10
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
trainer = Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train(net, train_iter, test_iter, batch_size, trainer, num_epochs)


"""
epoch 1, loss 0.5388, train acc 0.811, test acc 0.852, time 2641.7 sec
epoch 2, loss 0.3133, train acc 0.886, test acc 0.887, time 2572.8 sec
epoch 3, loss 0.2632, train acc 0.904, test acc 0.900, time 2573.3 sec
epoch 4, loss 0.2313, train acc 0.916, test acc 0.907, time 2660.2 sec
epoch 5, loss 0.2082, train acc 0.924, test acc 0.901, time 2691.3 sec
epoch 6, loss 0.1924, train acc 0.930, test acc 0.897, time 2669.2 sec
epoch 7, loss 0.1776, train acc 0.936, test acc 0.922, time 2669.5 sec
epoch 8, loss 0.1643, train acc 0.941, test acc 0.897, time 2605.2 sec
epoch 9, loss 0.1518, train acc 0.944, test acc 0.903, time 2584.2 sec
epoch 10, loss 0.1414, train acc 0.949, test acc 0.930, time 2584.0 sec
"""





















