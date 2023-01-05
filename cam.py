import os
import numpy as np

from PIL import Image
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision.transforms import transforms
from BAM import BAM
from CBAM import CBAM
from CA import CA
from TA import TripletAttention
from TCA import TCA
#-----------------------------------------------MobileNetV2的可视化---------------------------------------------------------------------#
def main():
    # 这个下面放置你网络的代码，因为载入权重的时候需要读取网络代码，这里我建议直接从自己的训练代码中原封不动的复制过来即可，我这里因为跑代码使用的是Resnet，所以这里将resent的网络复制到这里即可
    def Conv3x3BNReLU(in_channels, out_channels, stride, groups):
        return nn.Sequential(
            # stride=2 wh减半，stride=1 wh不变
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                      groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    # PW卷积
    def Conv1x1BNReLU(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    # # PW卷积(Linear) 没有使用激活函数
    def Conv1x1BN(in_channels, out_channels):
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
            out = (out + self.shortcut(x)) if self.stride == 1 else out
            # 第二种:
            # out = (out + x) if self.stride == 1 and self.in_channels == self.out_channels else out
            return out

    num_class = 100

    class MobileNetV2(nn.Module):
        def __init__(self, num_classes=num_class, t=6):
            super(MobileNetV2, self).__init__()
            self.first_conv = Conv3x3BNReLU(3, 32, 2, groups=1)
            #self.ca = CA(inp=32, oup=32)
            #self.ta = TripletAttention()
            #self.tca = TCA()
            self.layer1 = self.make_layer(in_channels=32, out_channels=16, stride=1, factor=1, block_num=1)
            self.layer2 = self.make_layer(in_channels=16, out_channels=24, stride=2, factor=t, block_num=2)
            self.layer3 = self.make_layer(in_channels=24, out_channels=32, stride=2, factor=t, block_num=3)
            self.layer4 = self.make_layer(in_channels=32, out_channels=64, stride=2, factor=t, block_num=4)
            self.layer5 = self.make_layer(in_channels=64, out_channels=96, stride=1, factor=t, block_num=3)
            self.layer6 = self.make_layer(in_channels=96, out_channels=160, stride=2, factor=t, block_num=3)
            self.layer7 = self.make_layer(in_channels=160, out_channels=320, stride=1, factor=t, block_num=1)
            self.last_conv = Conv1x1BNReLU(320, 1280)
            self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
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
            x = self.first_conv(x)  # torch.Size([1, 32, 112, 112])
            #x = self.ca(x)
            #x = self.ta(x)
            #x = self.tca(x)
            x = self.layer1(x)  # torch.Size([1, 16, 112, 112])
            x = self.layer2(x)  # torch.Size([1, 24, 56, 56])
            x = self.layer3(x)  # torch.Size([1, 32, 28, 28])
            x = self.layer4(x)  # torch.Size([1, 64, 14, 14])
            x = self.layer5(x)  # torch.Size([1, 96, 14, 14])
            x = self.layer6(x)  # torch.Size([1, 160, 7, 7])
            x = self.layer7(x)  # torch.Size([1, 320, 7, 7])
            x = self.last_conv(x)  # torch.Size([1, 1280, 7, 7])
            x = self.avgpool(x)  # torch.Size([1, 1280, 1, 1])
            x = x.view(x.size(0), -1)  # torch.Size([1, 1280])
            x = self.dropout(x)
            x = self.linear(x)  # torch.Size([1, 5])
            return x

    net = MobileNetV2()
#-----------------------------------------------Resnet50的可视化---------------------------------------------------------------------#
# def main():
#
#     device = torch.device('cuda')
#     class Bottleneck(nn.Module):
#         #每个stage维度中扩展的倍数
#         extention = 4
#         def __init__(self, inplanes, planes, stride, downsample=None):
#
#             '''
#             :param inplanes: 输入block的之前的通道数
#             :param planes: 在block中间处理的时候的通道数
#                     planes*self.extention:输出的维度
#             :param stride:
#             :param downsample:
#             '''
#             super(Bottleneck, self).__init__()
#
#             self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
#             self.bn1 = nn.BatchNorm2d(planes)
#             self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#             self.bn2 = nn.BatchNorm2d(planes)
#             #self.ca = CA(inp=planes, oup=planes)
#             #self.ta = TripletAttention()
#             self.tca = TCA()
#             self.conv3 = nn.Conv2d(planes, planes*self.extention, kernel_size=1, stride=1, bias=False)
#             self.bn3 = nn.BatchNorm2d(planes*self.extention)
#             self.relu = nn.ReLU(inplace=True)
#             #判断残差有没有卷积
#             self.downsample = downsample
#             self.stride = stride
#
#         def forward(self, x):
#             #参差数据
#             residual = x
#             #卷积操作
#             out = self.conv1(x)
#             out = self.bn1(out)
#             out = self.relu(out)
#             out = self.conv2(out)
#             out = self.bn2(out)
#             out = self.relu(out)
#             #加注意力
#             #out = self.ca(out)
#             #out = self.ta(out)
#             out = self.tca(out)
#             out = self.conv3(out)
#             out = self.bn3(out)
#             out = self.relu(out)
#             #是否直连（如果Indentity blobk就是直连；如果Conv2 Block就需要对残差边就行卷积，改变通道数和size
#             if self.downsample is not None:
#                 residual = self.downsample(x)
#             #将残差部分和卷积部分相加
#             out = out+residual
#             out = self.relu(out)
#             return out
#
#     class ResNet(nn.Module):
#         def __init__(self, block, layers, num_class):
#             #inplane=当前的fm的通道数
#             self.inplane = 64
#             super(ResNet, self).__init__()
#             #参数
#             self.block = block
#             self.layers = layers
#             #stem的网络层
#             self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=7, stride=2, padding=3, bias=False)
#             self.bn1 = nn.BatchNorm2d(self.inplane)
#             self.relu = nn.ReLU()
#             self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#             #64,128,256,512指的是扩大4倍之前的维度，即Identity Block中间的维度
#             self.stage1 = self.make_layer(self.block, 64, layers[0], stride=1)
#             self.stage2 = self.make_layer(self.block, 128, layers[1], stride=2)
#             self.stage3 = self.make_layer(self.block, 256, layers[2], stride=2)
#             self.stage4 = self.make_layer(self.block, 512, layers[3], stride=2)
#             #后续的网络
#             self.avgpool = nn.AvgPool2d(7)
#             self.fc = nn.Linear(512*block.extention, num_class)
#
#         def forward(self, x):
#             #stem部分：conv+bn+maxpool
#             out = self.conv1(x)
#             out = self.bn1(out)
#             out = self.relu(out)
#             out = self.maxpool(out)
#             #block部分
#             out = self.stage1(out)
#             out = self.stage2(out)
#             out = self.stage3(out)
#             out = self.stage4(out)
#             #分类
#             out = self.avgpool(out)
#             out = torch.flatten(out, 1)
#             out = self.fc(out)
#             return out
#
#         def make_layer(self, block, plane, block_num, stride=1):
#             '''
#             :param block: block模板
#             :param plane: 每个模块中间运算的维度，一般等于输出维度/4
#             :param block_num: 重复次数
#             :param stride: 步长
#             :return:
#             '''
#             block_list = []
#             #先计算要不要加downsample
#             downsample = None
#             if(stride!=1 or self.inplane!=plane*block.extention):
#                 downsample=nn.Sequential(
#                     nn.Conv2d(self.inplane, plane*block.extention, stride=stride, kernel_size=1, bias=False),
#                     nn.BatchNorm2d(plane*block.extention)
#                 )
#             # Conv Block输入和输出的维度（通道数和size）是不一样的，所以不能连续串联，他的作用是改变网络的维度
#             # Identity Block 输入维度和输出（通道数和size）相同，可以直接串联，用于加深网络
#             #Conv_block
#             conv_block = block(self.inplane, plane, stride=stride, downsample=downsample)
#             block_list.append(conv_block)
#             self.inplane = plane*block.extention
#             #Identity Block
#             for i in range(1, block_num):
#                 block_list.append(block(self.inplane, plane, stride=1))
#             return nn.Sequential(*block_list)
    device = torch.device('cuda')
    #net = ResNet(Bottleneck,[3,4,6,3],100).to(device)#50 resnet101[3,4,23,3] resnet152[3,8,36,3]
    net = net.to(device)
    net.load_state_dict(
        #------------------------MobileNetV2----------------------------------------------------
        torch.load("./model-10064-mobil-noatt", map_location=device))  # 载入训练的resnet模型权重，你将训练的模型权重放到当前文件夹下即可
        #torch.load("./model-10064-mobil-ca", map_location=device))  # 载入训练的resnet模型权重，你将训练的模型权重放到当前文件夹下即可
        #torch.load("./model-10064-mobil-ta", map_location=device))
        #torch.load("./model-10064-mobil-tca", map_location=device))
        #------------------------ResNet50----------------------------------------------------
        #torch.load("./resnet-save/model-10064-resnet-ca", map_location=device))
        #torch.load("./resnet-save/model-10064-resnet-ta", map_location=device))
        #torch.load("./resnet-save/model-10064-resnet-tca", map_location=device))
    target_layers = [net.last_conv]  # 这里是 看你是想看那一层的输出，我这里是打印的resnet最后一层的输出，你也可以根据需要修改成自己的
    print(target_layers)
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # 导入图片
    img_path = "./n0209960100000214.jpg"  # 这里是导入你需要测试图片
    image_size = 224  # 训练图像的尺寸，在你训练图像的时候图像尺寸是多少这里就填多少
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')  # 将图片转成RGB格式的
    img = np.array(img, dtype=np.uint8)  # 转成np格式
    img = center_crop_img(img, image_size)  # 将测试图像裁剪成跟训练图片尺寸相同大小的

    # [C, H, W]
    img_tensor = data_transform(img)  # 简单预处理将图片转化为张量
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)  # 增加一个batch维度
    cam = GradCAM(model=net, target_layers=target_layers, use_cuda=True)
    grayscale_cam = cam(input_tensor=input_tensor)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.savefig('./noatt-6.png')  # 将热力图的结果保存到本地当前文件夹
    plt.show()


if __name__ == '__main__':
    main()