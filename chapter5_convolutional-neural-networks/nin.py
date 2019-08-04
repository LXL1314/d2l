from charpter5 import *
from mxnet import init
from mxnet.gluon import nn, Trainer

def nin_block(num_channels, kernel_size, strides, padding):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(channels=num_channels, kernel_size=kernel_size, strides=strides,
                      padding=padding, activation='relu'),
            nn.Conv2D(channels=num_channels, kernel_size=1, activation='relu'),
            nn.Conv2D(channels=num_channels, kernel_size=1, activation='relu'))
    return blk

net = nn.Sequential()
net.add(nin_block(num_channels=96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2D(pool_size=3, strides=2),
        nin_block(num_channels=256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2D(pool_size=3, strides=2),
        nin_block(num_channels=384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2D(pool_size=3, strides=2), nn.Dropout(0.5),
        nin_block(num_channels=10, kernel_size=3, strides=1, padding=1),
        nn.GlobalAvgPool2D(),
        nn.Flatten())

batch_size = 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
net.initialize(init=init.Xavier())

lr, num_epochs,   = 0.1,  30
trainer = Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train(net, train_iter, test_iter, batch_size, trainer, num_epochs)