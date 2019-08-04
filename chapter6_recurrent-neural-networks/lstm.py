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

    W_xi, W_hi, b_i = _three()
    W_xf, W_hf, b_f = _three()
    W_xo, W_ho, b_o = _three()
    W_xc, W_hc, b_c = _three()
    W_hq, b_q = _one(shape=(num_hiddens, num_outputs)), nd.zeros(shape=(num_outputs,))

    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]

    for param in params:
        param.attach_grad()

    return params

##定义模型
def init_lstm_state(batch_size, num_hiddens):
    ## Ct 和 Ht
    return (nd.zeros(shape=(batch_size, num_hiddens)), nd.zeros(shape=(batch_size, num_hiddens)))

def lstm(inputs, state, params):
    #inputs 输入前已经经过one_hot
    (H, C)= state
    outputs = []
    W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q = params
    for X in inputs:
        I = nd.sigmoid(nd.dot(X, W_xi) + nd.dot(H, W_hi) + b_i)
        F = nd.sigmoid(nd.dot(X, W_xf) + nd.dot(H, W_hf) + b_f)
        O = nd.sigmoid(nd.dot(X, W_xo) + nd.dot(H, W_ho) + b_o)
        C_tilda = nd.tanh(nd.dot(X, W_xc) + nd.dot(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * nd.tanh(C)
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H, C)

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
def predict_lstm(prefix, num_chars, lstm, params, init_lstm_state,
                num_hiddens, vocab_size, idx_to_char, char_to_idx):
    state = init_lstm_state(1, num_hiddens)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = nd.one_hot(nd.array([output[-1]]), vocab_size)
        Y, state = lstm(X, state, params)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])

##训练函数
def train_and_predict_lstm(lstm, get_params, init_lstm_state, num_hiddens,
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
            state = init_lstm_state(batch_size, num_hiddens)
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx=None)
        for X, Y in data_iter:
            if is_random_iter:
                state = init_lstm_state(batch_size, num_hiddens)
            else:
                for s in state:
                    s.detach()
            with autograd.record():
                inputs = nd.one_hot(X.T, vocab_size)
                outputs, state = lstm(inputs, state, params)
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
                print("-:", predict_lstm(prefix, pred_len, lstm, params, init_lstm_state,
                                        num_hiddens, vocab_size, idx_to_char, char_to_idx))


num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 5, 50, ['分开', '不分开']
train_and_predict_lstm(lstm, get_params, init_lstm_state, num_hiddens,
                      vocab_size, corpus_indices, idx_to_char,
                      char_to_idx, False, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)

##运行时间大概4.1sec左右



















