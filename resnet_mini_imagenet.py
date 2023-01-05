import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torch
import torch.nn as nn
from torchstat import stat
import torchvision.models as models
import numpy as np
import time
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from CA import CA
from TA import TripletAttention
from TCA import TCA
from BAM import BAM
from CBAM import CBAM

# 判断是否有GPU
device = torch.device('cuda')
BATCH_SIZE = 64#（100epoch 最好不低于16）
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

#读取数据
dataset_train = datasets.ImageFolder('C:/Users/QiangHe/PycharmProjects/Classfication/train', transform_train)
dataset_test = datasets.ImageFolder('C:/Users/QiangHe/PycharmProjects/Classfication/test', transform_test)
#dataset_val = datasets.ImageFolder('data/val', transform)

# 上面这一段是加载测试集的
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)#训练集
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)#测试集

#______________________________________________________model_________________________________________________________________________#
class Bottleneck(nn.Module):
    #每个stage维度中扩展的倍数
    extention = 4
    def __init__(self, inplanes, planes, stride, downsample=None):

        '''
        :param inplanes: 输入block的之前的通道数
        :param planes: 在block中间处理的时候的通道数
                planes*self.extention:输出的维度
        :param stride:
        :param downsample:
        '''
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.tca = TCA()
        #self.ca = CA(inp=planes, oup=planes)
        #self.ta = TripletAttention()
        #self.bam = BAM(channel=planes)
        #self.cbam = CBAM(channel=planes)
        self.conv3 = nn.Conv2d(planes, planes*self.extention, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.extention)
        self.relu = nn.ReLU(inplace=True)
        #判断残差有没有卷积
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        #参差数据
        residual = x
        #卷积操作
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        #加注意力
        #out = self.ca(out)
        out = self.tca(out)
        #out = self.ta(out)
        #out = self.bam(out)
        #out = self.cbam(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        #是否直连（如果Indentity blobk就是直连；如果Conv2 Block就需要对残差边就行卷积，改变通道数和size
        if self.downsample is not None:
            residual = self.downsample(x)
        #将残差部分和卷积部分相加
        out = out+residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_class):
        #inplane=当前的fm的通道数
        self.inplane = 64
        super(ResNet, self).__init__()
        #参数
        self.block = block
        self.layers = layers
        #stem的网络层
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #64,128,256,512指的是扩大4倍之前的维度，即Identity Block中间的维度
        self.stage1 = self.make_layer(self.block, 64, layers[0], stride=1)
        self.stage2 = self.make_layer(self.block, 128, layers[1], stride=2)
        self.stage3 = self.make_layer(self.block, 256, layers[2], stride=2)
        self.stage4 = self.make_layer(self.block, 512, layers[3], stride=2)
        #后续的网络
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512*block.extention, num_class)

    def forward(self, x):
        #stem部分：conv+bn+maxpool
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        #block部分
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        #分类
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def make_layer(self, block, plane, block_num, stride=1):
        '''
        :param block: block模板
        :param plane: 每个模块中间运算的维度，一般等于输出维度/4
        :param block_num: 重复次数
        :param stride: 步长
        :return:
        '''
        block_list = []
        #先计算要不要加downsample
        downsample = None
        if(stride!=1 or self.inplane!=plane*block.extention):
            downsample=nn.Sequential(
                nn.Conv2d(self.inplane, plane*block.extention, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(plane*block.extention)
            )
        # Conv Block输入和输出的维度（通道数和size）是不一样的，所以不能连续串联，他的作用是改变网络的维度
        # Identity Block 输入维度和输出（通道数和size）相同，可以直接串联，用于加深网络
        #Conv_block
        conv_block = block(self.inplane, plane, stride=stride, downsample=downsample)
        block_list.append(conv_block)
        self.inplane = plane*block.extention
        #Identity Block
        for i in range(1, block_num):
            block_list.append(block(self.inplane, plane, stride=1))
        return nn.Sequential(*block_list)
model = ResNet(Bottleneck,[3,4,6,3],100).to(device)#50 resnet101[3,4,23,3] resnet152[3,8,36,3]
#______________________________________________________train_________________________________________________________________________#
# 损失函数
criterion = nn.CrossEntropyLoss()
learning_rate = 0.01  # 初始学习率0.1
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.001)#0.0001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
#余弦退火
# scheduler = CosineAnnealingLR(optimizer, T_max=10)#
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
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
            # writer1.add_scalar('train-loss', loss.item(), epoch)
            if (i + 1) % 150 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        scheduler.step()
        # model.eval()
        # validate(test_loader, model, criterion)
        # writer2.add_scalar('train-top1', validate(test_loader, model, criterion)[0], epoch)
        # writer3.add_scalar('train-top5', validate(test_loader, model, criterion)[1], epoch)
#______________________________________________________valid_________________________________________________________________________#
# test
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
#______________________________________________________main_________________________________________________________________________#
# def params_count(model):
#     return np.sum([p.numel() for p in model.parameters()]).item()

if __name__ == '__main__':
        # torch.cuda.synchronize()
        # start = time.time()

        print(model)
        # stat(model, (3, 224, 224))
        # params_count(model)
        # total_loss = SummaryWriter("./logg")
        # writer1 = SummaryWriter(log_dir='./logg/Loss-cbam')  # 实例化writer
        # # 测试损失（训练损失+测试损失）的变化
        # writer2 = SummaryWriter(log_dir='./logg/top1-cbam')  # 实例化writer
        # writer3 = SummaryWriter(log_dir='./logg/top5-cbam')  # 实例化writer
        train()
        # writer1.close()
        # writer2.close()
        # writer3.close()
        #调用模型
        #model.load_state_dict(torch.load("./resnet-save/model-10064-resnet-tca_new9"))
        validate(test_loader, model, criterion)
        # Save the model
        torch.save(model.state_dict(), './resnet-save/model-10064-resnet-tca')

        # torch.cuda.synchronize()
        # end = time.time()
        # print("time:", time.localtime(start), time.localtime(end))