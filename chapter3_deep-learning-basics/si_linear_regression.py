from mxnet import autograd, nd
from mxnet.gluon import data as gdata
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn
from mxnet import init
from mxnet import gluon
num_festures = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
training_data = nd.random.normal(scale=1, shape=(num_examples, num_festures))
labels = true_w[0]*training_data[:, 0] + true_w[1]*training_data[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

batch_size = 10
dataset = gdata.ArrayDataset(training_data, labels)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(init.Normal(sigma=0.01))

loss = gloss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

num_epoches = 3
for epoch in range(1, num_epoches + 1):
    for X,y in data_iter:
        with autograd.record():
            lo = loss(net(X), y)
        lo.backward()
        trainer.step(batch_size)
    lo = loss(net(training_data), labels)
    print('epoch %d, loss: %f' % (epoch, lo.mean().asnumpy()))
dense = net[0]
print('true_w', true_w)
print('w', dense.weight.data())
print('true_b', true_b)
print('bias', dense.bias.data())
print('bias.grad', dense.bias.grad)








