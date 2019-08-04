import mxnet as mx
from mxnet import autograd, gluon, image, init, nd
from mxnet.gluon import data as gdata, loss as gloss, utils as gutils
from fig_config import *
import sys
import time

set_figsize()
img = image.imread('../img/cat1.jpg')
plt.imshow(img.asnumpy())
plt.show()





