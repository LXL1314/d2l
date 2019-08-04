from charpter5 import *
from mxnet import init
from mxnet.gluon import nn, Trainer

def vgg_block(net, num_convs, num_channels):
    #blk = nn.Sequential()
    for _ in range(num_convs):
        net.add(nn.Conv2D(num_channels, kernel_size=3, padding=1, activation='relu'))
    net.add(nn.MaxPool2D(pool_size=2, strides=2))
    return net

def vgg(conv_arch):
    net = nn.Sequential()
    for (num_convs, num_channels) in conv_arch:
        net = vgg_block(net, num_convs, num_channels)
    net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(10))
    return net

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
net = vgg(conv_arch)
net.initialize(init=init.Xavier())
lr, num_epochs, batch_size = 0.01, 30, 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
trainer = Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train(net, train_iter, test_iter, batch_size, trainer, num_epochs)


