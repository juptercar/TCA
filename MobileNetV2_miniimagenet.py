from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torch
import torch.nn as nn
import random
import pandas as pd
import matplotlib.pyplot as plt
import time
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import shutil
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
from torchstat import stat
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from BAM import BAM
from CBAM import CBAM
from CA import CA
from TA import TripletAttention
from TCA import TCA

# 判断是否有GPU
device = torch.device('cuda')
BATCH_SIZE = 64#64（100epoch 最好不低于16）
num_epochs = 100
#______________________________________________________data_________________________________________________________________________#
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),#将shape为(H, W, C)的nump.ndarray或img转为shape为(C, H, W)的tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 归一化处理
 # 需要更多数据预处理，自己查
])
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 归一化处理
 # 需要更多数据预处理，自己查
])

# #读取数据
dataset_train = datasets.ImageFolder('C:/Users/QiangHe/PycharmProjects/Classfication/train', transform_train)
dataset_test = datasets.ImageFolder('C:/Users/QiangHe/PycharmProjects/Classfication/test', transform_test)
# #dataset_val = datasets.ImageFolder('data/val', transform)
#
# # 上面这一段是加载测试集的
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=1) # 训练集
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=1) # 测试集
# #val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True) # 验证集
# #print(len(train_loader)*num_epochs)

#______________________________________________________model_________________________________________________________________________#

# 分类个数
num_class = 100

# DW卷积
def Conv3x3BNReLU(in_channels,out_channels,stride,groups):
    return nn.Sequential(
            # stride=2 wh减半，stride=1 wh不变
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

# PW卷积
def Conv1x1BNReLU(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

# # PW卷积(Linear) 没有使用激活函数
def Conv1x1BN(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

class InvertedResidual(nn.Module):
    # t = expansion_factor,也就是扩展因子，文章中取6
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channels = (in_channels * expansion_factor)

        # 先1x1卷积升维，再1x1卷积降维

        self.bottleneck = nn.Sequential(
            # 升维操作: 扩充维度是 in_channels * expansion_factor (6倍)
            Conv1x1BNReLU(in_channels, mid_channels),
            # DW卷积,降低参数量
            Conv3x3BNReLU(mid_channels, mid_channels, stride, groups=mid_channels),
            # 降维操作: 降维度 in_channels * expansion_factor(6倍) 降维到指定 out_channels 维度
            Conv1x1BN(mid_channels, out_channels)
        )

        # 第一种: stride=1 才有shortcut 此方法让原本不相同的channels相同
        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)

        # 第二种: stride=1 切 in_channels=out_channels 才有 shortcut
        # if self.stride == 1 and in_channels == out_channels:
        #     self.shortcut = ()

    def forward(self, x):
        out = self.bottleneck(x)
        # 第一种:
        out = (out+self.shortcut(x)) if self.stride == 1 else out
        # 第二种:
        # out = (out + x) if self.stride == 1 and self.in_channels == self.out_channels else out
        return out

class MobileNetV2(nn.Module):
    # num_class为分类个数, t为扩充因子
    def __init__(self, num_classes=num_class, t=6):
        super(MobileNetV2, self).__init__()

        # 3 -> 32 groups=1 不是组卷积 单纯的卷积操作
        self.first_conv = Conv3x3BNReLU(3, 32, 2, groups=1)
        #self.ca = CA(inp=32, oup=32)
        #self.ta = TripletAttention()
        #self.tca = TCA()
        #self.bam = BAM(channel=32)
        #self.cbam = CBAM(channel=32)
        # 32 -> 16 stride=1 wh不变
        self.layer1 = self.make_layer(in_channels=32, out_channels=16, stride=1, factor=1, block_num=1)
        # 16 -> 24 stride=2 wh减半
        self.layer2 = self.make_layer(in_channels=16, out_channels=24, stride=2, factor=t, block_num=2)
        # 24 -> 32 stride=2 wh减半
        self.layer3 = self.make_layer(in_channels=24, out_channels=32, stride=2, factor=t, block_num=3)
        # 32 -> 64 stride=2 wh减半
        self.layer4 = self.make_layer(in_channels=32, out_channels=64, stride=2, factor=t, block_num=4)
        # 64 -> 96 stride=1 wh不变
        self.layer5 = self.make_layer(in_channels=64, out_channels=96, stride=1, factor=t, block_num=3)
        # 96 -> 160 stride=2 wh减半
        self.layer6 = self.make_layer(in_channels=96, out_channels=160, stride=2, factor=t, block_num=3)
        # 160 -> 320 stride=1 wh不变
        self.layer7 = self.make_layer(in_channels=160, out_channels=320, stride=1, factor=t, block_num=1)
        # 320 -> 1280 单纯的升维操作
        self.last_conv = Conv1x1BNReLU(320, 1280)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(in_features=1280, out_features=num_classes)
        self.init_params()

    def make_layer(self, in_channels, out_channels, stride, factor, block_num):
        layers = []
        # 与ResNet类似，每层Bottleneck单独处理，指定stride。此层外的stride均为1
        layers.append(InvertedResidual(in_channels, out_channels, factor, stride))
        # 这些叠加层stride均为1，in_channels = out_channels, 其中 block_num-1 为重复次数
        for i in range(1, block_num):
            layers.append(InvertedResidual(out_channels, out_channels, factor, 1))
        return nn.Sequential(*layers)

    # 初始化权重操作
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)  # torch.Size([1, 32, 112, 112])
        #x = self.ca(x)
        #x = self.ta(x)
        #x = self.cbam(x)
        #x = self.tca(x)
        #x = self.bam(x)
        x = self.layer1(x)      # torch.Size([1, 16, 112, 112])
        x = self.layer2(x)      # torch.Size([1, 24, 56, 56])
        x = self.layer3(x)      # torch.Size([1, 32, 28, 28])
        x = self.layer4(x)      # torch.Size([1, 64, 14, 14])
        x = self.layer5(x)      # torch.Size([1, 96, 14, 14])
        x = self.layer6(x)      # torch.Size([1, 160, 7, 7])
        x = self.layer7(x)      # torch.Size([1, 320, 7, 7])
        x = self.last_conv(x)   # torch.Size([1, 1280, 7, 7])
        x = self.avgpool(x)     # torch.Size([1, 1280, 1, 1])
        x = x.view(x.size(0), -1)    # torch.Size([1, 1280])
        x = self.dropout(x)
        x = self.linear(x)      # torch.Size([1, 5])
        return x
model = MobileNetV2().to(device)
#______________________________________________________train_________________________________________________________________________#
# 损失函数
criterion = nn.CrossEntropyLoss()
learning_rate = 0.01  # 初始学习率0.1
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.001)#0.0001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
#余弦退火
# scheduler = CosineAnnealingLR(optimizer, T_max=10)#
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

def train():
    total_step = len(train_loader)
    for epoch in range(num_epochs):

        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #writer1.add_scalar('train-loss', loss.item(), epoch)
            if (i + 1) % 150 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        #scheduler.step()

        #validate(test_loader, model, criterion)
        #writer2.add_scalar('train-top1', validate(test_loader, model, criterion)[0], epoch)
        #writer3.add_scalar('train-top5', validate(test_loader, model, criterion)[1], epoch)
#______________________________________________________valid_________________________________________________________________________#
# test
#创建test_acc.csv和var_acc.csv文件，记录top-1和top-5
# df = pd.DataFrame(columns=['time', 'step', 'train Loss', 'training accuracy'])#列名
# df.to_csv("./train_acc.csv", index=False) #路径可以根据需要更改
def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    val_loss = 0
    total = 0
    correct = 0
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.to(device)
            target = target.to(device)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            val_loss +=loss.item()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            val_loss += loss.item()
        print('Testing: Top1: {top1.avg:.4f}'.format(top1=top1))
        print('Testing: top5: {top5.avg:.4f} '.format(top5=top5))
    return top1.avg, top5.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res
#______________________________________________________params________________________________________________________________________#
def params_count(model):
    return np.sum([p.numel() for p in model.parameters()]).item()
#______________________________________________________main_________________________________________________________________________#
if __name__ == '__main__':
        torch.cuda.synchronize()
        start = time.time()
        print(model)
        # stat(model, (3, 224, 224))
        # params_count(model)
        # logs
        #total_loss = SummaryWriter("./log")
        #writer1 = SummaryWriter(log_dir='./log/Loss-TA')  # 实例化writer
        # 测试损失（训练损失+测试损失）的变化
        #writer2 = SummaryWriter(log_dir='./log/top1-TA')  # 实例化writer
        #writer3 = SummaryWriter(log_dir='./log/top5-TA')  # 实例化writer
        train()
        #writer1.close()
        #writer2.close()
        #writer3.close()
        # history = model.fit(
        #     transform_train, model(transform_train), batch_size=32, epochs=10, verbose=1, callbacks=None,
        #     validation_split=0.0, validation_data=None, shuffle=True,
        #     class_weight=None, sample_weight=None, initial_epoch=0
        # )
        #history.history  # 打印训练记录
        # #调用模型
        #model.load_state_dict(torch.load("./model-10064-mobil-bam"))
        validate(test_loader, model, criterion)
        #Save the model
        #torch.save(model.state_dict(), './model-10064-mobil-tca')

        torch.cuda.synchronize()
        end = time.time()
        print("time:", time.localtime(start), time.localtime(end))