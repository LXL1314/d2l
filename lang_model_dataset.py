import mxnet
from mxnet import nd, autograd
from mxnet.gluon import loss as gloss
from data_config import *
import random
import zipfile
import time
import math


def load_data_jay_lyrics():
    with zipfile.ZipFile('jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size

## 随机采样
def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_idx = list(range(num_examples))
    random.shuffle(example_idx)

    def _data(ops):
        return corpus_indices[ops: ops + num_steps]

    for i in range(epoch_size):
        i = i * batch_size
        batch_idx = example_idx[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_idx]
        Y = [_data(j * num_steps + 1) for j in batch_idx]
        yield  nd.array(X, ctx), nd.array(Y, ctx)

#相邻采样
def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    corpus_indices = nd.array(corpus_indices)
    batch_len = len(corpus_indices) // batch_size
    epochs = (batch_len - 1) // num_steps
    indices = corpus_indices[0: batch_size*batch_len].reshape([batch_size, batch_len])
    for i in range(epochs):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + 1 + num_steps]
        yield X, Y











