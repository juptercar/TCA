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
import math
from torchstat import stat
import numpy as np
from models import *
from BAM import BAM
from CBAM import CBAM
from CA import CA
from TA import TripletAttention
from TCA import TCA

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 100
#______________________________________________________data_________________________________________________________________________#
# 1. 载入并标准化 CIFAR10 数据
# 1. Load and normalizing the CIFAR10 training and test datasets using torchvision
# data augmentation 数据增强
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transforms_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#______________________________________________________net_________________________________________________________________________#
net, model_name = VGG(vgg_name='VGG11'), 'VGG11'
#net, model_name = ResNet18(), 'ResNet18'
#net, model_name = ResNet34(), 'ResNet34'
# net, model_name = MobileNetV2(), 'MobileNetV2'
net = net.to(device)
#______________________________________________________train_________________________________________________________________________#
# 损失函数
criterion = nn.CrossEntropyLoss()
learning_rate = 0.01  # 初始学习率0.1
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

def train():
    total_step = len(trainloader)
    for epoch in range(num_epochs):

        net.train()
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = net(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #writer1.add_scalar('train-loss', loss.item(), epoch)
            if i % 50 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
#______________________________________________________valid_________________________________________________________________________#
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
def params_count(net):
    return np.sum([p.numel() for p in net.parameters()]).item()
#______________________________________________________main_________________________________________________________________________#
if __name__ == '__main__':
        torch.cuda.synchronize()
        start = time.time()
        print(net)

        train()
        validate(testloader, net, criterion)
        #Save the model
        torch.save(net.state_dict(), 'cifar/net-vgg-tca-channel')
        torch.cuda.synchronize()
        end = time.time()
        print("time:", time.localtime(start), time.localtime(end))