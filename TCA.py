import torch
import torch.nn as nn

#ca模块 triplet旋转后的三个分支分经过坐标注意力机制
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x+3)/6#return self.relu(x + 3)/6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)

#Z-pool全局最大池化和平均池化将chw变为2*hw
class HWPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class Gate1(nn.Module):
    def __init__(self):
        super(Gate1, self).__init__()
        self.HW_pool = HWPool()#2*h*w
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))#2*h*1
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))#2*1*w
        self.conv = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(1)
        self.act = h_swish()#relu+sigmoid*self
        self.conv_h = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(1)
        self.bn3 = nn.BatchNorm2d(1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x = self.HW_pool(x)#将c变为2
        x_h = self.pool_h(x)#2h1
        x_w = self.pool_w(x).permute(0, 1, 3, 2)#2w1
        y = torch.cat([x_h, x_w], dim=2)#2(h+w)1
        y = self.conv(y)#1(h+w)1
        y = self.bn1(y)#1(h+w)1
        y = self.act(y)#1(h+w)1
        x_h, x_w = torch.split(y, [h, w], dim=2)#1h1 1w1
        x_w = x_w.permute(0, 1, 3, 2)#11w

        a_h = self.conv_h(x_h)
        a_h = self.bn2(a_h)
        a_h = self.relu1(a_h)
        a_h = self.sigmoid1(a_h)
        a_w = self.conv_w(x_w)
        a_w = self.bn3(a_w)
        a_w = self.relu2(a_w)
        a_w = self.sigmoid2(a_w)
        out =identity * a_h *a_w
        #out = identity * y
        return out

# class Gate2(nn.Module):
#     def __init__(self):
#         super(Gate2, self).__init__()
#         self.HW_pool = HWPool()
#         #self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
#         self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
#         self.bn = nn.BatchNorm2d(1)
#         self.relu = nn.ReLU()
#         self.sigmod = nn.Sigmoid()
#
#     def forward(self, x):
#         out = self.HW_pool(x)
#         out = self.conv(out)
#         out = self.bn(out)#(归一化)
#         out = self.relu(out)
#         return out * self.sigmod(out)
class Gate2(nn.Module):
    def __init__(self):
        super(Gate2, self).__init__()
        self.HW_pool = HWPool()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(1)
        )
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        i = x
        out = self.HW_pool(x)
        out = self.conv(out)
        return i * self.sigmod(out)

class TCA(nn.Module):
    def __init__(self):
        super(TCA, self).__init__()
        self.hw = Gate1()
        self.cw = Gate2()
        self.hc = Gate2()

    def forward(self, x):
        x_out1 = self.hw(x)
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out2 = self.cw(x_perm1)
        x_out2 = x_out2.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out3 = self.hc(x_perm2)
        x_out3 = x_out3.permute(0, 3, 2, 1).contiguous()
        return 1/3 * (x_out1 + x_out2 + x_out3)
        ######################channel############################
        # x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        # x_out2 = self.cw(x_perm1)
        # x_out2 = x_out2.permute(0, 2, 1, 3).contiguous()
        # x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        # x_out3 = self.hc(x_perm2)
        # x_out3 = x_out3.permute(0, 3, 2, 1).contiguous()
        # return 1/2 * (x_out2 + x_out3)
        ######################spatial############################
        # x_out1 = self.hw(x)
        # return x_out1
    # 串联效果加点和并联差不多
    # def forward(self, x):
    #     out = self.hw(x)
    #     x_perm1 = out.permute(0, 2, 1, 3).contiguous()
    #     out = self.cw(x_perm1)
    #     out = out.permute(0, 2, 1, 3).contiguous()
    #     x_perm2 = out.permute(0, 3, 2, 1).contiguous()
    #     out = self.hc(x_perm2)
    #     out = out.permute(0, 3, 2, 1).contiguous()
    #     return out

# if __name__=='__main__':
#     x = torch.rand(1, 64, 224, 224)
#     model = TCA_new1()
#     print(model)
#     out = model(x)
#     print(out.shape)