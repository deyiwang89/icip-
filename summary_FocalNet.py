import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from dataset_loader import MyData, MyTestData
# from model import RGBNet,FocalNet
from clstm import ConvLSTM
from functions import imsave
import argparse
from trainer import Trainer
from utils.evaluateFM import get_FM

import os

from torchsummary import summary


class FocalNet(nn.Module):
    def __init__(self, n_class=2):
        super(FocalNet, self).__init__()

        # original image's size = 256*256*3

        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2    2 layers

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4   2 layers

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.conv3_4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_4 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_4 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8   4 layers

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.conv4_4 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16      4 layers

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.conv5_4 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_4 = nn.ReLU(inplace=True)  # 1/32    4 layers

        # conv5 add
        self.conv5_c = nn.Conv2d(512, 64, 3, padding=1)

        # conv3 add
        self.conv4_c = nn.Conv2d(512, 64, 3, padding=1)

        # conv3 add
        self.conv3_c = nn.Conv2d(256, 64, 3, padding=1)

        # conv2_add
        self.conv2_c = nn.Conv2d(128, 64, 3, padding=1)

        # down sample
        self.down = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # m.weight.data.zero_()
                nn.init.normal(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x

        h = self.relu1_1(self.bn1_1(self.conv1_1(h)))
        h = self.relu1_2(self.bn1_2(self.conv1_2(h)))
        h_nopool1 = h
        h = self.pool1(h)
        pool1 = h

        h = self.relu2_1(self.bn2_1(self.conv2_1(h)))
        h = self.relu2_2(self.bn2_2(self.conv2_2(h)))
        h_nopool2 = h
        h = self.pool2(h)
        pool2 = h

        h = self.relu3_1(self.bn3_1(self.conv3_1(h)))
        h = self.relu3_2(self.bn3_2(self.conv3_2(h)))
        h = self.relu3_3(self.bn3_3(self.conv3_3(h)))
        h = self.relu3_4(self.bn3_4(self.conv3_4(h)))
        h_nopool3 = h
        h = self.pool3(h)
        pool3 = h

        h = self.relu4_1(self.bn4_1(self.conv4_1(h)))
        h = self.relu4_2(self.bn4_2(self.conv4_2(h)))
        h = self.relu4_3(self.bn4_3(self.conv4_3(h)))
        h = self.relu4_4(self.bn4_4(self.conv4_4(h)))
        h_nopool4 = h
        h = self.pool4(h)
        pool4 = h

        h = self.relu5_1(self.bn5_1(self.conv5_1(h)))
        h = self.relu5_2(self.bn5_2(self.conv5_2(h)))
        h = self.relu5_3(self.bn5_3(self.conv5_3(h)))
        h = self.relu5_4(self.bn5_4(self.conv5_4(h)))
        pool5 = h

        # side out
        pool1 = self.down(pool1)
        # print(pool1.size())

        pool2 = self.conv2_c(pool2)
        pool2 = pool2 + pool1
        # print(pool2.size())

        pool3 = self.conv3_c(pool3)
        pool3 = F.upsample(pool3, scale_factor=2, mode='bilinear')
        # print(pool3.size())

        pool4 = self.conv4_c(pool4)
        pool4 = F.upsample(pool4, scale_factor=2, mode='bilinear')
        pool4 = F.upsample(pool4, scale_factor=2, mode='bilinear')
        # print(pool4.size())

        pool5 = self.conv5_c(pool5)
        pool5 = F.upsample(pool5, scale_factor=2, mode='bilinear')
        pool5 = F.upsample(pool5, scale_factor=2, mode='bilinear')
        pool5 = pool5 + pool4 +pool3
        # print(pool5.size())

        return pool1, pool2, pool3, pool4, pool5

def FocalNet_net():
    model = FocalNet(n_class=1)
    return model

if __name__ == "__main__":

    with torch.no_grad():

        net = FocalNet_net()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = net.to(device)

        summary(model,(3, 256, 256))