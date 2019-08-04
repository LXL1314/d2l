import fashion_mnist as fm
import data_config as dc
import fig_config as fc
from mxnet import autograd, init, gluon, nd
from mxnet.gluon import loss as gloss, data as gdata, nn

n_train, n_test, num_features = 20, 100, 200
true_w, true_b = nd.ones((num_features, 1)) * 0.01, 0.05
data = nd.random.normal(shape=(n_train + n_test, num_features))
labels = nd.dot(data, true_w) + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
train_data, test_data = data[:n_train, :], data[n_train:, :]
train_labels, test_labels = labels[: n_train], labels[n_train:]

def init_params():
    w = nd.random.normal(scale=1, shape=(num_features, 1))
    b = nd.zeros(shape=(1,))
    w.attach_grad()
    b.attach_grad()
    return [w, b]

def l2_penalty(w):
    return (w**2).sum() / 2 #(1,)的矢量

batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = dc.linearRegression, dc.squared_loss
traindataset = gdata.ArrayDataset(train_data, train_labels)
train_iter = gdata.DataLoader(traindataset, batch_size, shuffle=True)

def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l.backward()
            dc.SGD([w, b], lr, batch_size)
        train_ls.append(loss(net(train_data, w, b), train_labels).mean().asscalar())
        test_ls.append(loss(net(test_data, w, b), test_labels).mean().asscalar())
    fc.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', w.norm().asscalar())

fit_and_plot(lambd=30)


