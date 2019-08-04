from lang_model_dataset import *
from mxnet import nd, autograd
from mxnet.gluon import loss as gloss
import time, math

##载入数据集
corpus_indices, char_to_idx, idx_to_char, vocab_size = load_data_jay_lyrics()

##one_hot 向量： inputs: (batch_size, num_steps), X = nd.one_hot(inputs.T, vocab_size) :(num_steps, batch_size, vocab_size)

##初始化模型参数
def get_params(num_inputs, num_hiddens, num_outputs):
    def _one(shape):
        return nd.random.normal(scale=0.01, shape=shape)

    W_xh = _one(shape=(num_inputs, num_hiddens))
    W_hh = _one(shape=(num_hiddens, num_hiddens))
    b_h = nd.zeros(shape=(num_hiddens,))

    W_hq = _one(shape=(num_hiddens, num_outputs))
    b_q = nd.zeros(shape=(num_outputs,))

    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()

    return params

num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size

##定义模型
def init_rnn_state(batch_size, num_hiddens):
    return nd.zeros(shape=(batch_size, num_hiddens))

def rnn(inputs, state, params):
    # inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    H = state
    outputs = []
    for X in inputs:
        #H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        H = nd.relu(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, H


##预测函数
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, idx_to_char, char_to_idx):
    H = init_rnn_state(1, num_hiddens)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = nd.one_hot(nd.array([output[-1]]), vocab_size)
        Y, H = rnn(X, H, params)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])


##裁剪梯度
def grad_clipping(params, theta):
    norm = nd.array([0])
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta/norm


##训练函数
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = data_iter_random
    else:
        data_iter_fn = data_iter_consecutive

    params = get_params(num_inputs, num_hiddens, num_outputs)
    loss = gloss.SoftmaxCrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:##
            state = init_rnn_state(batch_size, num_hiddens)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx=None)
        for X, Y in data_iter:
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens)
            else:  # 相邻采样；需要使用detach函数从计算图分离隐藏状态???为什么需要这步
                for s in state:
                    s.detach()
            #state = init_rnn_state(batch_size, num_hiddens)
            with autograd.record():
                inputs = nd.one_hot(X.T, vocab_size)
                outputs, state = rnn(inputs, state, params)
                outputs = nd.concat(*outputs, dim=0)
                y = Y.T.reshape((-1,))
                lo = loss(outputs, y).mean()
            lo.backward()
            grad_clipping(params, clipping_theta)
            SGD(params, lr, 1)#lo已经除以了y.size(也就是mean())，所以这里的第三个参数batch_size = 1
            l_sum += lo.asscalar() * y.size
            n += y.size

        if (epoch + 1) % pred_period == 0:
            print("epoch: %d, perplexity %f, time %.2f sec" % (epoch + 1, math.exp(l_sum/n), time.time() - start))
            for prefix in prefixes:
                print("-:", predict_rnn(prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, idx_to_char, char_to_idx))


num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 5, 50, ['分开', '不分开']
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, corpus_indices, idx_to_char,
                      char_to_idx, False, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)
##运行时间大概1.60sec左右

##1 调超参数
##2 在训练函数中去掉梯度裁剪：会在训练函数中的预测语句中出现 OverflowError: math range error的错误
##  裁剪梯度可以应对梯度爆炸，但无法应对梯度衰减问题。
##  通常由于这个原因，循环神经网络在实际中较难捕捉时间序列中时间步距离较大的依赖关系
##  门控循环神经网络的提出，正是为了更好地捕捉时间序列中时间步距离较大的依赖关系。
##3 将pred_period变量设为1
##4 将相邻采样改为不从计算图分离隐藏状态，运行时间有没有变化::: 感觉变化不明显？？
##5 将本节中使用的激活函数替换成ReLU，重复本节的实验

##从计算图分离隐含状态： 在模型训练的每次迭代中，当前批量序列的初始隐含状态来自上一个相邻批量序列的输出隐含状态。
##为了使模型参数的梯度计算只依赖当前的批量序列，从而减小每次迭代的计算开销，我们可以使用detach函数来将隐含状态从计算图中分离出来。

##将其分离出来和重新对其进行初始化有什么区别吗？
##



