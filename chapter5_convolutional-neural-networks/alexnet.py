from mxnet import init
from mxnet.gluon import nn, Trainer
from charpter5 import *

##定义模型
net = nn. Sequential()
net.add(nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5), ##防止过拟合
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5), ##防止过拟合
        nn.Dense(10))

batch_size = 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.01, 30
net.initialize(init=init.Xavier())
trainer = Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train(net, train_iter, test_iter, batch_size, trainer, num_epochs)

"""
epoch 1, loss 1.2964, train acc 0.515, test acc 0.759, time 3533.8 sec
epoch 2, loss 0.6534, train acc 0.756, test acc 0.804, time 3413.7 sec
epoch 3, loss 0.5407, train acc 0.798, test acc 0.830, time 3378.2 sec
epoch 4, loss 0.4758, train acc 0.824, test acc 0.851, time 3362.6 sec
epoch 5, loss 0.4340, train acc 0.843, test acc 0.859, time 3357.1 sec
epoch 6, loss 0.4019, train acc 0.853, test acc 0.870, time 3356.3 sec
epoch 7, loss 0.3782, train acc 0.863, test acc 0.878, time 3360.4 sec
epoch 8, loss 0.3603, train acc 0.869, test acc 0.882, time 3358.5 sec
epoch 9, loss 0.3437, train acc 0.874, test acc 0.889, time 3356.6 sec
epoch 10, loss 0.3304, train acc 0.881, test acc 0.886, time 3362.7 sec
epoch 11, loss 0.3186, train acc 0.884, test acc 0.897, time 3363.7 sec
epoch 12, loss 0.3079, train acc 0.888, test acc 0.897, time 3360.2 sec
epoch 13, loss 0.3011, train acc 0.889, test acc 0.901, time 3359.4 sec
epoch 14, loss 0.2911, train acc 0.894, test acc 0.897, time 3360.1 sec
epoch 15, loss 0.2831, train acc 0.898, test acc 0.902, time 3364.7 sec
epoch 16, loss 0.2748, train acc 0.900, test acc 0.902, time 3359.5 sec
epoch 17, loss 0.2686, train acc 0.902, test acc 0.908, time 3363.4 sec
epoch 18, loss 0.2615, train acc 0.904, test acc 0.907, time 3367.2 sec
epoch 19, loss 0.2565, train acc 0.906, test acc 0.905, time 3366.1 sec
epoch 20, loss 0.2513, train acc 0.907, test acc 0.908, time 3368.3 sec
epoch 21, loss 0.2431, train acc 0.909, test acc 0.909, time 3370.7 sec
epoch 22, loss 0.2366, train acc 0.912, test acc 0.911, time 3362.5 sec
epoch 23, loss 0.2350, train acc 0.914, test acc 0.906, time 3361.9 sec
epoch 24, loss 0.2269, train acc 0.916, test acc 0.914, time 3396.5 sec
"""


