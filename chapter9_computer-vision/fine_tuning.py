from mxnet import gluon, init
from mxnet.gluon import data as gdata, loss as gloss, model_zoo
from mxnet.gluon import utils as gutils, Trainer
from utils import train, try_all_gpus
import os
import zipfile

data_dir = '../data'
base_url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/'
fname = gutils.download(
    base_url + 'gluon/dataset/hotdog.zip',
    path=data_dir, sha1_hash='fba480ffa8aa7e0febbb511d181409f899b9baa5')
with zipfile.ZipFile(fname, 'r') as z:
    z.extractall(data_dir)


train_imgs = gdata.vision.ImageFolderDataset(
    os.path.join(data_dir, 'hotdog/train'))
test_imgs = gdata.vision.ImageFolderDataset(
    os.path.join(data_dir, 'hotdog/test'))


# 指定RGB三个通道的均值和方差来将图像通道归一化
normalize = gdata.vision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = gdata.vision.transforms.Compose([
    gdata.vision.transforms.RandomResizedCrop(224),
    ##从图像中裁剪出随机大小和随机高宽比的一块随机区域，将该区域缩放为高和宽均为224像素的输入
    gdata.vision.transforms.RandomFlipLeftRight(),##左右翻转，一半概率的图像左右翻转
    gdata.vision.transforms.ToTensor(),##通过ToTensor实例将图像数据从uint8格式变换成32位浮点数格式,并除以255使得所有像素的数值均在0到1之间
    normalize])

"""
Converts an image NDArray of shape (H x W x C) in the range
    [0, 255] to a float32 tensor NDArray of shape (C x H x W) in
    the range [0, 1).
"""

test_augs = gdata.vision.transforms.Compose([
    gdata.vision.transforms.Resize(256),##图像的高和宽均缩放为256像素
    gdata.vision.transforms.CenterCrop(224),##裁剪出高和宽均为224像素的中心区域作为输入
    gdata.vision.transforms.ToTensor(),##通过ToTensor实例将图像数据从uint8格式变换成32位浮点数格式,并除以255使得所有像素的数值均在0到1之间
    normalize])


##定义和初始化模型
pretrained_net = model_zoo.vision.resnet18_v2(pretrained=True)
finetune_net = model_zoo.vision.resnet18_v2(classes=2)##与预训练的源模型一样，但最后的输出个数等于目标数据集的类别数
finetune_net.features = pretrained_net.features##新模型除输出层外的层的模型参数初始化为预训练模型中除输出层外的层的参数模型
finetune_net.output.initialize(init.Xavier())##新模型输出层进行随机初始化
finetune_net.output.collect_params().setattr('lr_mult', 10)##以设定的学习率的10倍从头训练目标模型的输出层参数


def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5):
    train_iter = gdata.DataLoader(
        train_imgs.transform_first(train_augs), batch_size, shuffle=True)
    test_iter = gdata.DataLoader(
        test_imgs.transform_first(test_augs), batch_size)
    ctx = try_all_gpus()
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    loss = gloss.SoftmaxCrossEntropyLoss()
    trainer = Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': 0.001})##Trainer的学习率设得小一点，如0.01，对预训练得到的模型参数进行微调
    train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)

train_fine_tuning(finetune_net, 0.01)


"""
epoch 1, loss 3.0762, train acc 0.705, test acc 0.881, time 395.1 sec
epoch 2, loss 0.3246, train acc 0.905, test acc 0.909, time 391.2 sec
epoch 3, loss 0.9255, train acc 0.861, test acc 0.920, time 390.9 sec
epoch 4, loss 0.2481, train acc 0.929, test acc 0.897, time 391.1 sec
epoch 5, loss 0.2196, train acc 0.931, test acc 0.891, time 389.1 sec
"""



