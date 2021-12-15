#env:
# python 3.x
# pytorch 1.5.0
# torchvision 0.6.0
# cudatoolkit 10.2.89
# cudnn 7.6.5
# Actually you can configure them easily in anaconda

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms

#定义残差块，Resnet18一共有18层，分成5个大层，其中第一大层不带残差功能，由一个卷积层和一个池化层构成
#从第三层到第十八层，一共十六层，每两层构成一个残差块，因此一共有8个残差块，每一个大层两个
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        #我们为这个类传入了pytorch事先设定好的Module类，可以方便的调用接口构建模型
        #super(Net, self).init(),Net是父类，这里实际上是子类调用父类的init函数初始化整个类
        #这里的父类是ResidualBlock，也就是pytorch帮我们写好的ResidualBlock，我们借用一下它的init函数
        super(ResidualBlock, self).__init__()

        #.Sequential,用来构建模型，可以看出一个Sequential里面是卷积层-归一化-激活函数-卷积层-归一化
        #self.left是残差块的主要结构，主要结构像普通rnn那样计算，加上shortcut处理过的x(残差)相加之后就是这一个大层的真正输出
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        #shortcut,和resnet的原理有关，指将前面的部分输出跳过多个层级直接加权合并到后面的输出上去
        self.shortcut = nn.Sequential()
        #如果步长不为1或者输入输出通道数不一致，我们需要调整维度到合适的位置
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            
    #两个卷积层处理过后的结果和shortcut处理过的x值相加就是这个残差块的输出
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

#Resnet的主结构
class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=8):
        #老样子，用父类的init初始化这个类
        super(ResNet, self).__init__()
        #输入通道为64
        self.inchannel = 64
        #第一个卷积层，这里没加池化层，后面加上就可以了
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        #layer1其实是第二个大层，输入通道64
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)        
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        #全连接层，输入为512通道的1*1，其实也就是512个数，输出我们人为设定的类别
        self.fc = nn.Linear(4608, num_classes)
        
    #用这个函数来构建一个大层，输入使用的bolck，输入通道数，每一大层block数和步长即可
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    #网络前向传播过程,x是我们输入的图片,最后返回一个out，out是一个向量，反映了最终预测的结果
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18():
    return ResNet(ResidualBlock)