import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torch
import torch.nn as nn
from torchstat import stat
import numpy as np
import torch.nn.functional as F
from TCA import TCA
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion*planes)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion * planes)
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out
#
# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_planes = 64
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.tca = TCA()
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512*block.expansion, num_classes)
#
#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.tca(out)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out
#
# def ResNet34():
#     return ResNet(BasicBlock, [3, 4, 6, 3])
#
# model=ResNet34()
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

        self.bottleneck = nn.Sequential(
            Conv1x1BNReLU(in_channels, mid_channels),
            Conv3x3BNReLU(mid_channels, mid_channels, stride, groups=mid_channels),
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
num_class = 10
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=num_class, t=6):
        super(MobileNetV2, self).__init__()
        #self.tca = TCA()
        self.first_conv = Conv3x3BNReLU(3, 32, 2, groups=1)

        self.layer1 = self.make_layer(in_channels=32, out_channels=16, stride=1, factor=1, block_num=1)
        self.layer2 = self.make_layer(in_channels=16, out_channels=24, stride=2, factor=t, block_num=2)
        self.layer3 = self.make_layer(in_channels=24, out_channels=32, stride=2, factor=t, block_num=3)
        self.layer4 = self.make_layer(in_channels=32, out_channels=64, stride=2, factor=t, block_num=4)
        self.layer5 = self.make_layer(in_channels=64, out_channels=96, stride=1, factor=t, block_num=3)
        self.layer6 = self.make_layer(in_channels=96, out_channels=160, stride=1, factor=t, block_num=3)
        self.layer7 = self.make_layer(in_channels=160, out_channels=320, stride=1, factor=t, block_num=1)
        self.last_conv = Conv1x1BNReLU(320, 1280)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=1)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(in_features=1280, out_features=num_classes)
        self.init_params()

    def make_layer(self, in_channels, out_channels, stride, factor, block_num):
        layers = []
        layers.append(InvertedResidual(in_channels, out_channels, factor, stride))
        for i in range(1, block_num):
            layers.append(InvertedResidual(out_channels, out_channels, factor, 1))
        return nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #x = self.tca(x)
        x = self.first_conv(x)  # torch.Size([1, 32, 112, 112])

        x = self.layer1(x)      # torch.Size([1, 16, 16, 16])
        #print(x.size())
        x = self.layer2(x)      # torch.Size([1, 24, 8, 8])
        #print(x.size())
        x = self.layer3(x)      # torch.Size([1, 32, 4, 4])
        #print(x.size())
        x = self.layer4(x)      # torch.Size([1, 64, 2, 2])
        #print(x.size())
        x = self.layer5(x)      # torch.Size([1, 96, 2, 2])
        #print(x.size())
        x = self.layer6(x)      # torch.Size([1, 160, 2, 2])
        #print(x.size())
        x = self.layer7(x)      # torch.Size([1, 320, 2, 2 ])
        #print(x.size())
        x = self.last_conv(x)   # torch.Size([1, 1280, 7, 7])
        x = self.avgpool(x)     # torch.Size([1, 320, 1, 1])
        #print(x.size())
        x = x.view(x.size(0), -1)    # torch.Size([1, 1280])
        x = self.dropout(x)
        x = self.linear(x)      # torch.Size([1, 5])
        return x
model = MobileNetV2()
def params_count(model):
    return np.sum([p.numel() for p in model.parameters()]).item()

if __name__ == '__main__':
        print(model)
        stat(model, (3, 32, 32))
        params_count(model)
