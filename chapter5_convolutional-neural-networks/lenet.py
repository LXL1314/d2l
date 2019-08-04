from mxnet import gluon, init
from mxnet.gluon import nn
from fashion_mnist import load_data_fashion_mnist
from charpter5 import train

##定义模型
net1 = nn.Sequential()
net1.add(nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120, activation='relu'),
        nn.Dense(84, activation='relu'),
        nn.Dense(10))


##定义模型(使用了批量归一化）
net2 = nn.Sequential()
net2.add(nn.Conv2D(6, kernel_size=5), nn.BatchNorm(), nn.Activation('relu'),
         nn.MaxPool2D(pool_size=2, strides=2),
         nn.Conv2D(16, kernel_size=5), nn.BatchNorm(), nn.Activation('relu'),
         nn.MaxPool2D(pool_size=2, strides=2),
         nn.Dense(120), nn.BatchNorm(), nn.Activation('relu'),
         nn.Dense(84), nn.BatchNorm(), nn.Activation('relu'),
         nn.Dense(10))


 ##获取数据
batch_size, lr, num_epochs = 256, 0.1, 30
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

net1.initialize(init=init.Xavier())
trainer1 = gluon.Trainer(net1.collect_params(), 'sgd', {'learning_rate': lr})
train(net1, train_iter, test_iter, batch_size, trainer1, num_epochs)

net2.initialize(init=init.Xavier())
trainer2 = gluon.Trainer(net2.collect_params(), 'sgd', {'learning_rate': lr})
train(net2, train_iter, test_iter, batch_size, trainer2, num_epochs)

