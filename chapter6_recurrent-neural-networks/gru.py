from mxnet import nd
from mxnet.gluon import rnn, loss as gloss
from lang_model_dataset import *

##读取数据集
corpus_indices, char_to_idx, idx_to_char, vocab_size = load_data_jay_lyrics()

##初始化模型参数
def get_params(num_inputs, num_hiddens, num_outputs):
    def _one(shape):
        return nd.random.normal(scale=0.01, shape=shape)

    def _three():
        return (_one(shape=(num_inputs, num_hiddens)),
                _one(shape=(num_hiddens, num_hiddens)),
                nd.zeros(shape=(num_hiddens,)))

    (W_xz, W_hz, b_z) = _three()#更新门参数
    (W_xr, W_hr, b_r) = _three()#重置门参数
    (W_xh, W_hh, b_h) = _three()#候选隐藏状态参数
    W_hq = _one(shape=(num_hiddens, num_outputs))
    b_q = nd.zeros(shape=(num_outputs,))

    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]

    for param in params:
        param.attach_grad()

    return params

##定义模型
def init_gru_state(batch_size, num_hiddens):
    return nd.zeros(shape=(batch_size, num_hiddens))

def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H = state
    outputs = []
    for X in inputs:
        R = nd.sigmoid(nd.dot(X, W_xr) + nd.dot(H, W_hr) + b_r)
        Z = nd.sigmoid(nd.dot(X, W_xz) + nd.dot(H, W_hz) + b_z)
        H_tilda = nd.tanh(nd.dot(X, W_xh) + nd.dot(R*H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, H

##裁剪梯度
def grad_clipping(params, theta):
    norm = nd.array([0])
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta/norm


num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size

##预测函数
def predict_rnn(prefix, num_chars, gru, params, init_gru_state,
                num_hiddens, vocab_size, idx_to_char, char_to_idx):
    H = init_gru_state(1, num_hiddens)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = nd.one_hot(nd.array([output[-1]]), vocab_size)
        Y, H = gru(X, H, params)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])

##训练函数
def train_and_predict_gru(gru, get_params, init_gru_state, num_hiddens,
                          vocab_size, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):

    if is_random_iter:
        data_iter_fn = data_iter_consecutive
    else:
        data_iter_fn = data_iter_random

    params = get_params(num_inputs, num_hiddens, num_outputs)
    loss = gloss.SoftmaxCrossEntropyLoss()

    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        if not is_random_iter:
            state = init_gru_state(batch_size, num_hiddens)
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx=None)
        for X, Y in data_iter:
            if is_random_iter:
                state = init_gru_state(batch_size, num_hiddens)
            else:
                for s in state:
                    s.detach()
            with autograd.record():
                inputs = nd.one_hot(X.T, vocab_size)
                outputs, state = gru(inputs, state, params)
                outputs = nd.concat(*outputs, dim=0)
                y = Y.T.reshape(shape=(-1,))
                lo = loss(outputs, y).mean()
            lo.backward()
            grad_clipping(params, clipping_theta)
            SGD(params, lr, 1)
            l_sum += lo.asscalar() * y.size
            n += y.size

        if (epoch + 1) % pred_period == 0:
            print("epoch: %d, perplexity %f, time %.2f sec" % (epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print("-:", predict_rnn(prefix, pred_len, gru, params, init_gru_state,
                                        num_hiddens, vocab_size, idx_to_char, char_to_idx))


num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 5, 50, ['分开', '不分开']
train_and_predict_gru(gru, get_params, init_gru_state, num_hiddens,
                      vocab_size, corpus_indices, idx_to_char,
                      char_to_idx, False, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)

##运行时间大概3.2sec左右





