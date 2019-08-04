from matplotlib import pyplot as plt
from mxnet import autograd, nd
from fig_config import set_figsize
from data_config import data_iter, linearRegression, squared_loss, SGD

## generating training data
num_features = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
data = nd.random.normal(scale=1, shape=(num_examples, num_features))
#print("data[:, 0].shape",data[:, 0].shape) #(m,)
labels = true_w[0] * data[:, 0] + true_w[1] * data[:, 1] + true_b#(m,)
labels += nd.random.normal(scale=0.01, shape=labels.shape)  #labels.shape : (m,)

set_figsize()
plt.scatter(data[:, 1].asnumpy(), labels.asnumpy(), 1)
plt.show()

w = nd.random.normal(scale=0.01, shape=(num_features, 1))
b = nd.zeros(shape=(1,))

w.attach_grad()
b.attach_grad()

batch_size = 15
lr = 0.05
num_epochs = 3
net = linearRegression
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, data, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)
            #print("loss.sum(): ", l.sum())
        l.backward() ## 求导 l.backward() 等价于 l.sum().backward()
        SGD([w, b], lr, batch_size)
    train_l = loss(net(data, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().asnumpy()))

print('true_w', true_w)
print('w', w)
print('true_b', true_b)
print('b', b)
















