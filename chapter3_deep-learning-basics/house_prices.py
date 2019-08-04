import pandas as pd
import numpy as np
from mxnet import autograd, init, nd, gluon
from mxnet.gluon import loss as gloss, data as gdata, nn

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

n_train, n_test = train_data.shape[0], test_data.shape[0]

print('train_data shape : ', train_data.shape)
print('test_data shape : ', test_data.shape)

all_data = pd.concat([train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]])


















