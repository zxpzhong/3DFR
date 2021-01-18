# -*- coding:UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
from math import floor
from .spp import *
from params import Args
import sys
import platform
if platform.python_version().split('.')[0] == '2':
    sys.path.append('./')
    from center_loss import CenterLoss
else:
    from Model.center_loss import CenterLoss
class Bottleneck(nn.Module):
    '''
    Inverted Residual Block
    '''
    def __init__(self, in_channels, out_channels, expansion_factor=6, kernel_size=3, stride=2):
        super(Bottleneck, self).__init__()
        if stride != 1 and stride != 2:
            raise ValueError('Stride should be 1 or 2')
        # Inverted Residual Block
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expansion_factor, 1, bias=False),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor,
                      kernel_size, stride, padding=int((kernel_size-1)/2),
                      groups=in_channels * expansion_factor, bias=False),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels * expansion_factor, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            # Linear Bottleneck，这里不接ReLU6
            # nn.ReLU6(inplace=True)
        )
        # Assumption based on previous ResNet papers: If the number of filters doesn't match,
        # there should be a conv1x1 operation.
        self.if_match_bypss = True if in_channels != out_channels else False
        # src ERROR 不管输入输出通道是否相同，都要在输出做一次卷积？？
        self.if_match_bypss = True
        if self.if_match_bypss:
            self.bypass_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        output = self.block(x)
        # print(output.shape)
        # print(x.shape)
        if self.if_match_bypss:
            return output + self.bypass_conv(x)
        else:
            return output + x
def conv_bn(input, output, stride):
    '''
    普通卷积模块（conv + bn + relu）
    :param input: 输入
    :param output: 输出
    :param stride: 步长
    :return: 普通卷积block
    '''
    return nn.Sequential(
        nn.Conv2d(input, output, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(output),
        # inplace，默认设置为False，表示新创建一个对象对其修改，也可以设置为True，表示直接对这个对象进行修改
        nn.LeakyReLU(inplace=True)
    )
def conv_bottleneck(input, output, stride):
    return Bottleneck(in_channels=input, out_channels=output, stride=stride)
class MobileNet_v2(nn.Module):
    '''
    MobileNet v2网络
    '''
    def __init__(self, num_classes=107):
        '''
        构造函数
        :param num_classes: 总类别数
        '''
        super(MobileNet_v2, self).__init__()
        self.spp = SpatialPyramidPool2D([1,2,4])
        self.num_classes = num_classes
        self.conv_bn_1 = conv_bn(3, 32, 1)
        self.pool_1 = nn.MaxPool2d(kernel_size=2)
        self.conv_bottleneck_2 = conv_bottleneck(32, 64, 1)
        self.pool_2 = nn.MaxPool2d(kernel_size=2)
        self.conv_bottleneck_3 = conv_bottleneck(64, 128, 1)
        self.pool_3 = nn.MaxPool2d(kernel_size=2)
        self.drop_3 = nn.Dropout(Args.keep_prob)
        self.conv_bottleneck_4 = conv_bottleneck(128, 256, 1)
        self.pool_4 = nn.MaxPool2d(kernel_size=1)
        self.drop_4 = nn.Dropout(Args.keep_prob)
        # self.conv_bottleneck_5 = conv_bottleneck(128, 256, 1)
        # self.pool_5 = nn.MaxPool2d(kernel_size=2)
        # self.drop_5 = nn.Dropout(Args.keep_prob)
        self.fc_5 = nn.Linear(5376, 256)
        self.drop_5 = nn.Dropout(Args.keep_prob)
        self.fc_6 = nn.Linear(256, self.num_classes)
        # self.drop_6 = nn.Dropout(Args.keep_prob)
        # initialize model parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.conv_bn_1(x)
        x = self.pool_1(x)
        x = self.conv_bottleneck_2(x)
        x = self.pool_2(x)
        x = self.conv_bottleneck_3(x)
        x = self.pool_3(x)
        # print(x.shape)
        x = self.conv_bottleneck_4(x)
        x = self.pool_4(x)
        # x = self.conv_bottleneck_5(x)
        # x = self.pool_5(x)
        # x = self.drop_4(x)
        # print(x.shape)
        # exit()
        x = self.spp(x)
        # print(x.shape)
        # exit()
        x = x.view(-1, 5376)
        # x = x.reshape(-1, 5 * 17 * 128)
        # x = x.resize(x.shape[0],5*17*128)
        # x = x.view(x.shape[0], 5 * 17 * 128)
        feature = self.fc_5(x)
        x = self.drop_5(feature)
        x = self.fc_6(x)
        # x = self.drop_6(x)
        x = F.log_softmax(x, dim=-1)
        # x = F.softmax(x, dim=-1)
        return x, feature
class MobileNet_v2_feature(nn.Module):
    '''
    MobileNet v2网络
    '''
    def __init__(self):
        '''
        构造函数
        :param num_classes: 总类别数
        '''
        super(MobileNet_v2_feature, self).__init__()
        self.spp = SpatialPyramidPool2D([1,2,4])
        self.conv_bn_1 = conv_bn(3, 32, 1)
        self.pool_1 = nn.MaxPool2d(kernel_size=2)
        self.conv_bottleneck_2 = conv_bottleneck(32, 64, 1)
        self.pool_2 = nn.MaxPool2d(kernel_size=2)
        self.conv_bottleneck_3 = conv_bottleneck(64, 128, 1)
        self.pool_3 = nn.MaxPool2d(kernel_size=2)
        self.drop_3 = nn.Dropout(Args.keep_prob)
        self.conv_bottleneck_4 = conv_bottleneck(128, 256, 1)
        self.pool_4 = nn.MaxPool2d(kernel_size=1)
        self.drop_4 = nn.Dropout(Args.keep_prob)
        # self.conv_bottleneck_5 = conv_bottleneck(128, 256, 1)
        # self.pool_5 = nn.MaxPool2d(kernel_size=2)
        # self.drop_5 = nn.Dropout(Args.keep_prob)
        self.fc_5 = nn.Linear(5376, 256)
        self.drop_5 = nn.Dropout(Args.keep_prob)
        # self.fc_6 = nn.Linear(256, self.num_classes)
        # self.drop_6 = nn.Dropout(Args.keep_prob)
        # initialize model parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.conv_bn_1(x)
        x = self.pool_1(x)
        x = self.conv_bottleneck_2(x)
        x = self.pool_2(x)
        x = self.conv_bottleneck_3(x)
        x = self.pool_3(x)
        # print(x.shape)
        x = self.conv_bottleneck_4(x)
        x = self.pool_4(x)
        # x = self.conv_bottleneck_5(x)
        # x = self.pool_5(x)
        # x = self.drop_4(x)
        # print(x.shape)
        # exit()
        x = self.spp(x)
        # print(x.shape)
        # exit()
        x = x.view(-1, 5376)
        # x = x.reshape(-1, 5 * 17 * 128)
        # x = x.resize(x.shape[0],5*17*128)
        # x = x.view(x.shape[0], 5 * 17 * 128)
        feature = self.fc_5(x)
        return feature
class MobileNet_v2_class(nn.Module):
    '''
    MobileNet v2网络
    '''
    def __init__(self ,num_classes=107):
        '''
        构造函数
        :param num_classes: 总类别数
        '''
        super(MobileNet_v2_class, self).__init__()
        self.num_classes = num_classes
        self.drop_5 = nn.Dropout(Args.keep_prob)
        self.fc_6 = nn.Linear(256, self.num_classes)
        # initialize model parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.drop_5(x)
        x = self.fc_6(x)
        x = F.log_softmax(x, dim=-1)
        return x
if __name__ == '__main__':
    net = MobileNet_v2(num_classes=107)
    summary(net, (3, 360, 360))
    # data = torch.rand((8, 3, 360, 360))
    # output, embed = net(data)
    # print('input: {}'.format(data.shape))
    # print('output: {}'.format(output.shape))
    # # print(output)
    #
    # # embed = net.get_embedding(data)
    # print('embedding: {}'.format(embed.shape))
    #
    # loss = CenterLoss(num_classes=107, feat_dim=256)
    # labels = torch.Tensor(np.random.randint(low=0, high=107, size=8)).long()
    # print(labels.shape)
    # loss_out = loss(embed, labels)
    # print(loss_out)
 