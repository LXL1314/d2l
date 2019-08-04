from lang_model_dataset import *
from mxnet import nd, autograd, init
from mxnet.gluon import loss as gloss, nn, rnn, Trainer
import time, math

##载入数据集
corpus_indices, char_to_idx, idx_to_char, vocab_size = load_data_jay_lyrics()

##one_hot 向量： inputs: (batch_size, num_steps), X = nd.one_hot(inputs.T, vocab_size) :(num_steps, batch_size, vocab_size)

##定义模型
class RNNModel(nn.Block):
    def __init__(self, num_hiddens, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        #rnn_layer = rnn.RNN(num_hiddens)
        #rnn_layer.initialize()
        #self.rnn = rnn_layer
        self.rnn = rnn.RNN(num_hiddens)
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        #inputs: (batch_size, num_steps)
        #X :(num_steps, batch_size, vocab_size)
        X = nd.one_hot(inputs.T, self.vocab_size)
        #Y: (num_steps, batch_size, num_hiddens)
        #state: (隐藏层个数， batch_size, num_hiddens)
        Y, state = self.rnn(X, state)
        #output: (num_steps*batch_size, num_hiddens)
        #在进行loss时，y:(batch_size, num_steps), 要进行这一步：y = y.T.reshape(shape=(-1,))
        output = self.dense(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


##预测函数
def predict_rnn_gluon(prefix, num_chars, model, idx_to_char,
                      char_to_idx):
    state = model.begin_state(batch_size=1)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = nd.array([output[-1]]).reshape(shape=[1, 1])#batch_size=1, num_steps=1
        (Y, state) = model(X, state)# Y.shape:  (1, 1027)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])


##梯度裁剪
def grad_clipping(params, theta):
    norm = nd.array([0])
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if theta < norm:
        for param in params:
            param.grad[:] *= theta / norm


##训练函数
def train_and_predict_rnn_gluon(model, vocab_size,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes):

    model.initialize(init=init.Normal(0.01))
    loss = gloss.SoftmaxCrossEntropyLoss()
    trainer = Trainer(model.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0, 'wd': 0})

    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None)##相邻采样
        state = model.begin_state(batch_size=batch_size)
        for X, Y in data_iter:
            ##
            for s in state:
                s.detach()
            with autograd.record():
                output, state = model(X, state)
                y = Y.T.reshape(shape=[-1,])
                lo = loss(output, y).mean()
            lo.backward()
            params = [p.data() for p in model.collect_params().values()]
            grad_clipping(params, clipping_theta)
            trainer.step(1)
            l_sum += lo.asscalar() * y.size
            n += y.size

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print("-:", predict_rnn_gluon(prefix, pred_len, model, idx_to_char, char_to_idx))



num_epochs, num_steps, num_hiddens, batch_size, lr, clipping_theta = 250, 35, 256, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 5, 50, ['分开', '不分开']
model = RNNModel(num_hiddens, vocab_size)
##在train_and_predict_rnn_gluon这个函数中， model会被初始化

train_and_predict_rnn_gluon(model, vocab_size,
                            corpus_indices, idx_to_char, char_to_idx,
                            num_epochs, num_steps, lr, clipping_theta,
                            batch_size, pred_period, pred_len, prefixes)

##运行时间大概0.56sec左右

"""
与上一节的实现进行比较。看看Gluon的实现是不是运行速度更快？如果你觉得差别明显，试着找找原因。

## 可能是前项计算部分导致的区别： 在gluon中，是进行矩阵运算，但是在scratch中是进行for循环

"""




"""
num_hiddens = 256
model = RNNModel(num_hiddens, vocab_size)
model.initialize()
res = predict_rnn_gluon('分开', 10, model, vocab_size, idx_to_char, char_to_idx)
print(res)
"""