import torch
import torch.nn as nn
from TCA import TCA
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        layers += [TCA()]
        for x in cfg :
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]

        return nn.Sequential(*layers)
# class VGG16(nn.Module):
#     def __init__(self, num_classes=10):
#         super(VGG16, self).__init__()
#         self.features = nn.Sequential(
#             #1
#
#             nn.Conv2d(3,64,kernel_size=3,padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             TCA(),
#             #2
#             nn.Conv2d(64,64,kernel_size=3,padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2,stride=2),
#             #3
#             nn.Conv2d(64,128,kernel_size=3,padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             #4
#             nn.Conv2d(128,128,kernel_size=3,padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2,stride=2),
#             #5
#             nn.Conv2d(128,256,kernel_size=3,padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             #6
#             nn.Conv2d(256,256,kernel_size=3,padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             #7
#             nn.Conv2d(256,256,kernel_size=3,padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2,stride=2),
#             #8
#             nn.Conv2d(256,512,kernel_size=3,padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             #9
#             nn.Conv2d(512,512,kernel_size=3,padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             #10
#             nn.Conv2d(512,512,kernel_size=3,padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2,stride=2),
#             #11
#             nn.Conv2d(512,512,kernel_size=3,padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             #12
#             nn.Conv2d(512,512,kernel_size=3,padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             #13
#             nn.Conv2d(512,512,kernel_size=3,padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2,stride=2),
#             nn.AvgPool2d(kernel_size=1,stride=1),
#             )
#         self.classifier = nn.Sequential(
#             #14
#             nn.Linear(512,4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             #15
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             #16
#             nn.Linear(4096,num_classes),
#             )
#         #self.classifier = nn.Linear(512, 10)
#
#     def forward(self, x):
#         out = self.features(x)
#         #        print(out.shape)
#         out = out.view(out.size(0), -1)
#         #        print(out.shape)
#         out = self.classifier(out)
#         #        print(out.shape)
#         return out