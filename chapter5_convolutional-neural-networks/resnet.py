from mxnet import init, nd
from mxnet.gluon import nn, Trainer
from charpter5 import *

class Residual(nn.Block):
    def __init__(self, num_channels, use_lxlConv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=strides)##(in+2-3)/strides + 1
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)##channel,大小与做完conv1一致

        if use_lxlConv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1, strides=strides)##(in+0-1)/strides + 1##channel,大小与做完conv1一致
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = nd.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return nd.relu(Y + X)

net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))

def resnet_block(num_channels, num_residuals, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            ##非第一个模块的模块中的第一个残差块需要使用1x1的卷积层改变通道数
            ##每个模块的第一个残差块要将高和宽减半，所以这里的strides=2
            blk.add(Residual(num_channels, use_lxlConv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk


net.add(resnet_block(64, 2, first_block=True),
        resnet_block(128, 2),
        resnet_block(256, 2),
        resnet_block(512, 2))

net.add(nn.GlobalAvgPool2D(), nn.Dense(10))
net.initialize(init=init.Xavier())

lr, num_epochs, batch_size = 0.05, 5, 256
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
trainer = Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train(net, train_iter, test_iter, batch_size, trainer, num_epochs)

##后续改进看了论文再来实现

"""
运行结果
epoch 1, loss 0.4847, train acc 0.830, test acc 0.885, time 4001.4 sec
epoch 2, loss 0.2597, train acc 0.903, test acc 0.902, time 3920.5 sec
epoch 3, loss 0.1974, train acc 0.927, test acc 0.896, time 3817.2 sec
epoch 4, loss 0.1496, train acc 0.946, test acc 0.914, time 3819.0 sec
epoch 5, loss 0.1153, train acc 0.959, test acc 0.882, time 3882.9 sec
"""
