from mxnet.gluon import loss as gloss, trainer, nn
from mxnet import init, gluon
import fashion_mnist as fm

batch_size = 256
train_iter, test_iter = fm.load_data_fashion_mnist(batch_size)
drop_prob1, drop_prob2 = 0.2, 0.5
drop_prob1, drop_prob2 = 0.0, 0.0

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dropout(drop_prob2),
        nn.Dense(256, activation='relu'),
        nn.Dropout(drop_prob1),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

loss = gloss.SoftmaxCrossEntropyLoss()
lr = 0.5
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})

num_epochs = 5
fm.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, trainer=trainer)






