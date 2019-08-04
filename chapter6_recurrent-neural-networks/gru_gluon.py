from lang_model_dataset import *
from mxnet import nd, autograd, init
from mxnet.gluon import loss as gloss, nn, rnn, Trainer
import time, math

##载入数据集
corpus_indices, char_to_idx, idx_to_char, vocab_size = load_data_jay_lyrics()

##定义模型
class GRUModel(nn.Block):
    def __init__(self, num_hiddens, vocab_size, **kwargs):
        super(GRUModel, self).__init__(**kwargs)
        self.gru = rnn.GRU(num_hiddens)
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        X = nd.one_hot(inputs.T, self.vocab_size)
        Y, state = self.gru(X, state)
        #output: num_steps * batch_size, vocab_size
        output = self.dense(Y.reshape(shape=(-1, Y.shape[-1])))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.gru.begin_state(*args, **kwargs)

##预测函数
def predict_gru_gluon(prefix, num_chars, model, idx_to_char, char_to_idx):
    state = model.begin_state(batch_size=1)
    outputs = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = nd.array([outputs[-1]]).reshape(shape=[1, 1])
        Y, state = model(X, state)
        if t < len(prefix) - 1:
            outputs.append(char_to_idx[prefix[t + 1]])
        else:
            outputs.append(int(Y.argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in outputs])

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
def train_and_predict_gru_gluon(model, corpus_indices, idx_to_char, char_to_idx,
                                is_random_iter, num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes):

    model.initialize(init=init.Normal(0.01))
    loss = gloss.SoftmaxCrossEntropyLoss()
    trainer = Trainer(model.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0, 'wd': 0})

    if is_random_iter:
        data_iter_fn = data_iter_random
    else:
        data_iter_fn = data_iter_consecutive

    for epoch in range(num_epochs):
        if not is_random_iter:
            state = model.begin_state(batch_size=batch_size)

        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, None)

        l_sum, n, start = 0.0, 0, time.time()

        for X, Y in data_iter:
            if is_random_iter:
                state = model.begin_state(batch_size)
            else:
                for s in state:
                    s.detach()

            with autograd.record():
                outputs, state = model(X, state)
                y = Y.T.reshape(shape=(-1,))
                lo = loss(outputs, y).mean()
                # loss(outputs, y)求的损失是单个样本损失组成的一个向量，形状是：（样本个数，）
                # （这里的样本总个数是y.size）

            lo.backward()
            params = [p.data() for p in model.collect_params().values()]
            grad_clipping(params, clipping_theta)
            trainer.step(1)

            l_sum += lo.asscalar() * y.size #所有单个样本loss的和
            n += y.size

        if (epoch + 1) % pred_period == 0:
            print("epoch: %d, perplexity %f, time %.2f sec" % (epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print("-:", predict_gru_gluon(prefix, pred_len, model, idx_to_char, char_to_idx))


num_epochs, num_hiddens, num_steps, batch_size, lr, clipping_theta = 250, 256, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 5, 50, ['分开', '不分开']
model = GRUModel(num_hiddens, vocab_size)

train_and_predict_gru_gluon(model, corpus_indices, idx_to_char, char_to_idx,
                                True, num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)


## 运行时间大概： 1.20sec 左右






