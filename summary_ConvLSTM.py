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
# from clstm import ConvLSTM
from functions import imsave
import argparse
from trainer import Trainer
from utils.evaluateFM import get_FM

import os

from torchsummary import summary


class ConvLSTM(nn.Module):
    def __init__(self,n_class=2):
        super(ConvLSTM, self).__init__()

        # ---------------------------- ConvLSTM1 ------------------------------ #

        # ------------------ ConvLSTM cell parameter ---------------------- #
        # self.conv_cell1 =  nn.Conv2d(64 + 64 , 4 * 64, 5, padding=2)
        self.conv_cell2 = nn.Conv2d(64 + 64, 4 * 64, 5, padding=2)
        self.conv_cell3 = nn.Conv2d(64 + 64, 4 * 64, 5, padding=2)
        self.conv_cell4 = nn.Conv2d(64 + 64, 4 * 64, 5, padding=2)
        self.conv_cell5 = nn.Conv2d(64 + 64, 4 * 64, 5, padding=2)

        # attentive convlstm 2
        self.conv_cell = nn.Conv2d(64 + 64, 4 * 64, 5, padding=2)



        # # 1st layer light-field features weighted

        # self.conv_w1 = nn.Conv2d(64 * 13, 13, 1, padding=0)  # 13 = 1+N (N=12)
        # self.pool_avg_w1 = nn.AvgPool2d(64, stride=2, ceil_mode=True)

        # 2nd layer light-field features weighted
        self.conv_w2 = nn.Conv2d(64 * 13, 13, 1, padding=0)  # 13 = 1+N (N=12)
        self.pool_avg_w2 = nn.AvgPool2d(64, stride=2, ceil_mode=True)
        # 3rd layer light-field features weighted
        self.conv_w3 = nn.Conv2d(64 * 13, 13, 1, padding=0)  # 13 = 1+N (N=12)
        self.pool_avg_w3 = nn.AvgPool2d(64, stride=2, ceil_mode=True)
        # 4th layer light-field features weighted
        self.conv_w4 = nn.Conv2d(64 * 13, 13, 1, padding=0)  # 13 = 1+N (N=12)
        self.pool_avg_w4 = nn.AvgPool2d(64, stride=2, ceil_mode=True)
        # 5th layer light-field features weighted
        self.conv_w5 = nn.Conv2d(64 * 13, 13, 1, padding=0)  # 13 = 1+N (N=12)
        self.pool_avg_w5 = nn.AvgPool2d(64, stride=2, ceil_mode=True)
        '''
        # -----------------------------  Multi-scale1  ----------------------------- #
        # part1:
        self.Atrous_c1 = nn.Conv2d(64, 64, 3, padding=1, dilation=1)  # size:  64*64*64
        self.Atrous_b1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r1 = nn.ReLU(inplace=True)
        # part2:
        self.Atrous_c3 = nn.Conv2d(64, 64, 3, padding=3, dilation=3)  # size:  64*64*64
        self.Atrous_b3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r3 = nn.ReLU(inplace=True)
        # part3:
        self.Atrous_c5 = nn.Conv2d(64, 64, 3, padding=5, dilation=5)  # size:  64*64*64
        self.Atrous_b5 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r5 = nn.ReLU(inplace=True)
        # part4:
        self.Atrous_c7 = nn.Conv2d(64, 64, 3, padding=7, dilation=7)  # size:  64*64*64
        self.Atrous_b7 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r7 = nn.ReLU(inplace=True)
        # conv
        self.Aconv = nn.Conv2d(64 * 5, 64, 1, padding=0)
        '''
        # -----------------------------  Multi-scale2  ----------------------------- #
        # part1:
        self.Atrous_c1_2 = nn.Conv2d(64, 64, 3, padding=1, dilation=1)  # size:  64*64*64
        self.Atrous_b1_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r1_2 = nn.ReLU(inplace=True)
        # part2:
        self.Atrous_c3_2 = nn.Conv2d(64, 64, 3, padding=3, dilation=3)  # size:  64*64*64
        self.Atrous_b3_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r3_2 = nn.ReLU(inplace=True)
        # part3:
        self.Atrous_c5_2 = nn.Conv2d(64, 64, 3, padding=5, dilation=5)  # size:  64*64*64
        self.Atrous_b5_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r5_2 = nn.ReLU(inplace=True)
        # part4:
        self.Atrous_c7_2 = nn.Conv2d(64, 64, 3, padding=7, dilation=7)  # size:  64*64*64
        self.Atrous_b7_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r7_2 = nn.ReLU(inplace=True)
        # conv
        self.Aconv_2 = nn.Conv2d(64 * 5, 64, 1, padding=0)

        # -----------------------------  Multi-scale3  ----------------------------- #
        # part1:
        self.Atrous_c1_3 = nn.Conv2d(64, 64, 3, padding=1, dilation=1)  # size:  64*64*64
        self.Atrous_b1_3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r1_3 = nn.ReLU(inplace=True)
        # part2:
        self.Atrous_c3_3 = nn.Conv2d(64, 64, 3, padding=3, dilation=3)  # size:  64*64*64
        self.Atrous_b3_3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r3_3 = nn.ReLU(inplace=True)
        # part3:
        self.Atrous_c5_3 = nn.Conv2d(64, 64, 3, padding=5, dilation=5)  # size:  64*64*64
        self.Atrous_b5_3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r5_3 = nn.ReLU(inplace=True)
        # part4:
        self.Atrous_c7_3 = nn.Conv2d(64, 64, 3, padding=7, dilation=7)  # size:  64*64*64
        self.Atrous_b7_3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r7_3 = nn.ReLU(inplace=True)
        # conv
        self.Aconv_3 = nn.Conv2d(64 * 5, 64, 1, padding=0)

        # -----------------------------  Multi-scale4  ----------------------------- #
        # part1:
        self.Atrous_c1_4 = nn.Conv2d(64, 64, 3, padding=1, dilation=1)  # size:  64*64*64
        self.Atrous_b1_4 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r1_4 = nn.ReLU(inplace=True)
        # part2:
        self.Atrous_c3_4 = nn.Conv2d(64, 64, 3, padding=3, dilation=3)  # size:  64*64*64
        self.Atrous_b3_4 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r3_4 = nn.ReLU(inplace=True)
        # part3:
        self.Atrous_c5_4 = nn.Conv2d(64, 64, 3, padding=5, dilation=5)  # size:  64*64*64
        self.Atrous_b5_4 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r5_4 = nn.ReLU(inplace=True)
        # part4:
        self.Atrous_c7_4 = nn.Conv2d(64, 64, 3, padding=7, dilation=7)  # size:  64*64*64
        self.Atrous_b7_4 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r7_4 = nn.ReLU(inplace=True)
        # conv
        self.Aconv_4 = nn.Conv2d(64 * 5, 64, 1, padding=0)

        # -----------------------------  Multi-scale5  ----------------------------- #
        # part1:
        self.Atrous_c1_5 = nn.Conv2d(64, 64, 3, padding=1, dilation=1)  # size:  64*64*64
        self.Atrous_b1_5 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r1_5 = nn.ReLU(inplace=True)
        # part2:
        self.Atrous_c3_5 = nn.Conv2d(64, 64, 3, padding=3, dilation=3)  # size:  64*64*64
        self.Atrous_b3_5 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r3_5 = nn.ReLU(inplace=True)
        # part3:
        self.Atrous_c5_5 = nn.Conv2d(64, 64, 3, padding=5, dilation=5)  # size:  64*64*64
        self.Atrous_b5_5 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r5_5 = nn.ReLU(inplace=True)
        # part4:
        self.Atrous_c7_5 = nn.Conv2d(64, 64, 3, padding=7, dilation=7)  # size:  64*64*64
        self.Atrous_b7_5 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r7_5 = nn.ReLU(inplace=True)
        # conv
        self.Aconv_5 = nn.Conv2d(64 * 5, 64, 1, padding=0)
 

        # ----------------------------- Attentive ConvLSTM 2 -------------------------- #
        # ConvLSTM-2
        # 2-1
        self.conv_fcn2_1 = nn.Conv2d(64, 64, 1, padding=0)  # 13 = 1+N (N=12)
        self.conv_h_1 = nn.Conv2d(64, 64, 1, padding=0)
        self.pool_avg_1 = nn.AvgPool2d(64, stride=2, ceil_mode=True)
        self.conv_c_1 = nn.Conv2d(64, 64, 1, padding=0)
        # 2-2
        self.conv_fcn2_2 = nn.Conv2d(64, 64, 1, padding=0)  # 13 = 1+N (N=12)
        self.conv_h_2 = nn.Conv2d(64, 64, 1, padding=0)
        self.pool_avg_2 = nn.AvgPool2d(64, stride=2, ceil_mode=True)
        self.conv_c_2 = nn.Conv2d(64, 64, 1, padding=0)
        # 2-3
        self.conv_fcn2_3 = nn.Conv2d(64, 64, 1, padding=0)  # 13 = 1+N (N=12)
        self.conv_h_3 = nn.Conv2d(64, 64, 1, padding=0)
        self.pool_avg_3 = nn.AvgPool2d(64, stride=2, ceil_mode=True)
        self.conv_c_3 = nn.Conv2d(64, 64, 1, padding=0)
        # 2-4
        self.conv_fcn2_4 = nn.Conv2d(64, 64, 1, padding=0)  # 13 = 1+N (N=12)
        self.conv_h_4 = nn.Conv2d(64, 64, 1, padding=0)
        self.pool_avg_4 = nn.AvgPool2d(64, stride=2, ceil_mode=True)
        self.conv_c_4 = nn.Conv2d(64, 64, 1, padding=0)
        # 2-5
        self.conv_fcn2_5 = nn.Conv2d(64, 64, 1, padding=0)  # 13 = 1+N (N=12)
        self.conv_h_5 = nn.Conv2d(64, 64, 1, padding=0)
        self.pool_avg_5 = nn.AvgPool2d(64, stride=2, ceil_mode=True)
        self.conv_c_5 = nn.Conv2d(64, 64, 1, padding=0)

        self.prediction = nn.Conv2d(64, 2, 1, padding=0)

        # ------------------- Middle Layer supervision -------------------- #
        self.prediction2 = nn.Conv2d(64, 2, 1, padding=0)
        self.prediction3 = nn.Conv2d(64, 2, 1, padding=0)
        self.prediction4 = nn.Conv2d(64, 2, 1, padding=0)
        self.prediction5 = nn.Conv2d(64, 2, 1, padding=0)

        # ------------------- ConvLSTM cell_out supervision -------------------- #
        self.pred_cell_out1 = nn.Conv2d(64, 2, 1, padding=0)
        self.pred_cell_out2 = nn.Conv2d(64, 2, 1, padding=0)
        self.pred_cell_out3 = nn.Conv2d(64, 2, 1, padding=0)


        # self._initialize_weights()

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #            # m.weight.data.zero_()
    #             nn.init.normal(m.weight.data, std=0.01)
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         if isinstance(m, nn.ConvTranspose2d):
    #             assert m.kernel_size[0] == m.kernel_size[1]
    #             initial_weight = get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
    #             m.weight.data.copy_(initial_weight)

    def convlstm_cell(self,A, new_c):
        (ai, af, ao, ag) = torch.split(A, A.size()[1] // 4, dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ag)
        g = torch.tanh(ag)
        new_c = f * new_c + i * g
        new_h = o * torch.tanh(new_c)
        return new_c , new_h



    def forward(self, level1):

        '''
        # ------------------- the 1st ConvLSTM layer ------------------ #
        # 1st layer light-field features weighted
        level1_ori = level1
        level1 = torch.cat(torch.chunk(level1, 13, dim=0), dim=1)
        weight1 = self.conv_w1(level1)
        weight1 = self.pool_avg_w1(weight1)
        weight1 = torch.mul(F.softmax(weight1, dim=1), 13)
        weight1 = weight1.transpose(0,1)
        level1 = torch.mul(level1_ori, weight1)
        '''

        # ------------------- the 2nd ConvLSTM layer ------------------ #
        # 2nd layer light-field features weighted
        #############----------------warning----------------#############
        level1 = torch.rand(13,64,64,64).float().cuda()
        level2 = level1
        level3 = level1
        level4 = level1
        level5 = level1
        level2_ori = level2
        #############----------------warning----------------#############
        level2 = torch.cat(torch.chunk(level2, 13, dim=0), dim=1)
        weight2 = self.conv_w2(level2)
        weight2 = self.pool_avg_w2(weight2)
        weight2 = torch.mul(F.softmax(weight2, dim=1), 13)
        weight2 = weight2.transpose(0, 1)
        level2 = torch.mul(level2_ori, weight2)

        # ------------------- the 3rd ConvLSTM layer ------------------ #
        # 3rd layer light-field features weighted
        level3_ori = level3
        level3 = torch.cat(torch.chunk(level3, 13, dim=0), dim=1)
        weight3 = self.conv_w3(level3)
        weight3 = self.pool_avg_w3(weight3)
        weight3 = torch.mul(F.softmax(weight3, dim=1), 13)
        weight3 = weight3.transpose(0, 1)
        level3 = torch.mul(level3_ori, weight3)

        # ------------------- the 4th ConvLSTM layer ------------------ #
        # 4th layer light-field features weighted
        level4_ori = level4
        level4 = torch.cat(torch.chunk(level4, 13, dim=0), dim=1)
        weight4 = self.conv_w4(level4)
        weight4 = self.pool_avg_w4(weight4)
        weight4 = torch.mul(F.softmax(weight4, dim=1), 13)
        weight4 = weight4.transpose(0, 1)
        level4 = torch.mul(level4_ori, weight4)

        # ------------------- the 5th ConvLSTM layer ------------------ #
        # 5th layer light-field features weighted
        level5_ori = level5
        level5 = torch.cat(torch.chunk(level5, 13, dim=0), dim=1)
        weight5 = self.conv_w5(level5)
        weight5 = self.pool_avg_w5(weight5)
        weight5 = torch.mul(F.softmax(weight5, dim=1), 13)
        weight5 = weight5.transpose(0, 1)
        level5 = torch.mul(level5_ori, weight5)


        # a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13 = torch.chunk(level1, 13, dim=0)	# level1 split
        b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13 = torch.chunk(level2, 13, dim=0)	# level2 split
        c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13 = torch.chunk(level3, 13, dim=0)	# level3 split
        d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13 = torch.chunk(level4, 13, dim=0)	# level4 split
        e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13 = torch.chunk(level5, 13, dim=0)	# level5 split

        '''
        # -----------------------------  level 1 spatial-temporal ConvLSTM  --------------------------------- #
        cell0 = a1
        h_state0 = a1
        # focal_slice1
        combined = torch.cat((a1, h_state0), dim=1)
        A = self.conv_cell1(combined)
        new_c, new_h = self.convlstm_cell(A,cell0)
        # focal_slice2
        combined = torch.cat((a2, new_h), dim=1)
        A = self.conv_cell1(combined)
        new_c, new_h = self.convlstm_cell(A,new_c)
        # focal_slice3
        combined = torch.cat((a3, new_h), dim=1)
        A = self.conv_cell1(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice4
        combined = torch.cat((a4, new_h), dim=1)
        A = self.conv_cell1(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice5
        combined = torch.cat((a5, new_h), dim=1)
        A = self.conv_cell1(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice6
        combined = torch.cat((a6, new_h), dim=1)
        A = self.conv_cell1(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice7
        combined = torch.cat((a7, new_h), dim=1)
        A = self.conv_cell1(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice8
        combined = torch.cat((a8, new_h), dim=1)
        A = self.conv_cell1(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice9
        combined = torch.cat((a9, new_h), dim=1)
        A = self.conv_cell1(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice10
        combined = torch.cat((a10, new_h), dim=1)
        A = self.conv_cell1(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice11
        combined = torch.cat((a11, new_h), dim=1)
        A = self.conv_cell1(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice12
        combined = torch.cat((a12, new_h), dim=1)
        A = self.conv_cell1(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice13
        combined = torch.cat((a13, new_h), dim=1)
        A = self.conv_cell1(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        out1 = new_h
        '''

        # -----------------------------  level 2 spatial-temporal ConvLSTM  --------------------------------- #
        cell0 = b1
        h_state0 = b1
        # focal_slice1
        combined = torch.cat((b1, h_state0), dim=1)
        A = self.conv_cell2(combined)
        new_c, new_h = self.convlstm_cell(A, cell0)
        # focal_slice2
        combined = torch.cat((b2, new_h), dim=1)
        A = self.conv_cell2(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice3
        combined = torch.cat((b3, new_h), dim=1)
        A = self.conv_cell2(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice4
        combined = torch.cat((b4, new_h), dim=1)
        A = self.conv_cell2(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice5
        combined = torch.cat((b5, new_h), dim=1)
        A = self.conv_cell2(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice6
        combined = torch.cat((b6, new_h), dim=1)
        A = self.conv_cell2(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice7
        combined = torch.cat((b7, new_h), dim=1)
        A = self.conv_cell2(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice8
        combined = torch.cat((b8, new_h), dim=1)
        A = self.conv_cell2(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice9
        combined = torch.cat((b9, new_h), dim=1)
        A = self.conv_cell2(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice10
        combined = torch.cat((b10, new_h), dim=1)
        A = self.conv_cell2(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice11
        combined = torch.cat((b11, new_h), dim=1)
        A = self.conv_cell2(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice12
        combined = torch.cat((b12, new_h), dim=1)
        A = self.conv_cell2(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice13
        combined = torch.cat((b13, new_h), dim=1)
        A = self.conv_cell2(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        out2 = new_h

        # -----------------------------  level 3 spatial-temporal ConvLSTM  --------------------------------- #
        cell0 = c1
        h_state0 = c1
        # focal_slice1
        combined = torch.cat((c1, h_state0), dim=1)
        A = self.conv_cell3(combined)
        new_c, new_h = self.convlstm_cell(A, cell0)
        # focal_slice2
        combined = torch.cat((c2, new_h), dim=1)
        A = self.conv_cell3(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice3
        combined = torch.cat((c3, new_h), dim=1)
        A = self.conv_cell3(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice4
        combined = torch.cat((c4, new_h), dim=1)
        A = self.conv_cell3(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice5
        combined = torch.cat((c5, new_h), dim=1)
        A = self.conv_cell3(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice6
        combined = torch.cat((c6, new_h), dim=1)
        A = self.conv_cell3(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice7
        combined = torch.cat((c7, new_h), dim=1)
        A = self.conv_cell3(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice8
        combined = torch.cat((c8, new_h), dim=1)
        A = self.conv_cell3(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice9
        combined = torch.cat((c9, new_h), dim=1)
        A = self.conv_cell3(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice10
        combined = torch.cat((c10, new_h), dim=1)
        A = self.conv_cell3(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice11
        combined = torch.cat((c11, new_h), dim=1)
        A = self.conv_cell3(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice12
        combined = torch.cat((c12, new_h), dim=1)
        A = self.conv_cell3(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice13
        combined = torch.cat((c13, new_h), dim=1)
        A = self.conv_cell3(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        out3 = new_h

        # -----------------------------  level 4 spatial-temporal ConvLSTM  --------------------------------- #
        cell0 = d1
        h_state0 = d1
        # focal_slice1
        combined = torch.cat((d1, h_state0), dim=1)
        A = self.conv_cell4(combined)
        new_c, new_h = self.convlstm_cell(A, cell0)
        # focal_slice2
        combined = torch.cat((d2, new_h), dim=1)
        A = self.conv_cell4(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice3
        combined = torch.cat((d3, new_h), dim=1)
        A = self.conv_cell4(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice4
        combined = torch.cat((d4, new_h), dim=1)
        A = self.conv_cell4(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice5
        combined = torch.cat((d5, new_h), dim=1)
        A = self.conv_cell4(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice6
        combined = torch.cat((d6, new_h), dim=1)
        A = self.conv_cell4(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice7
        combined = torch.cat((d7, new_h), dim=1)
        A = self.conv_cell4(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice8
        combined = torch.cat((d8, new_h), dim=1)
        A = self.conv_cell4(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice9
        combined = torch.cat((d9, new_h), dim=1)
        A = self.conv_cell4(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice10
        combined = torch.cat((d10, new_h), dim=1)
        A = self.conv_cell4(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice11
        combined = torch.cat((d11, new_h), dim=1)
        A = self.conv_cell4(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice12
        combined = torch.cat((d12, new_h), dim=1)
        A = self.conv_cell4(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice13
        combined = torch.cat((d13, new_h), dim=1)
        A = self.conv_cell4(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        out4 = new_h


        # -----------------------------  level 5 spatial-temporal ConvLSTM  --------------------------------- #
        cell0 = e1
        h_state0 = e1
        # focal_slice1
        combined = torch.cat((e1, h_state0), dim=1)
        A = self.conv_cell5(combined)
        new_c, new_h = self.convlstm_cell(A, cell0)
        # focal_slice2
        combined = torch.cat((e2, new_h), dim=1)
        A = self.conv_cell5(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice3
        combined = torch.cat((e3, new_h), dim=1)
        A = self.conv_cell5(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice4
        combined = torch.cat((e4, new_h), dim=1)
        A = self.conv_cell5(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice5
        combined = torch.cat((e5, new_h), dim=1)
        A = self.conv_cell5(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice6
        combined = torch.cat((e6, new_h), dim=1)
        A = self.conv_cell5(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice7
        combined = torch.cat((e7, new_h), dim=1)
        A = self.conv_cell5(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice8
        combined = torch.cat((e8, new_h), dim=1)
        A = self.conv_cell5(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice9
        combined = torch.cat((e9, new_h), dim=1)
        A = self.conv_cell5(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice10
        combined = torch.cat((e10, new_h), dim=1)
        A = self.conv_cell5(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice11
        combined = torch.cat((e11, new_h), dim=1)
        A = self.conv_cell5(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice12
        combined = torch.cat((e12, new_h), dim=1)
        A = self.conv_cell5(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        # focal_slice13
        combined = torch.cat((e13, new_h), dim=1)
        A = self.conv_cell5(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        out5 = new_h

        # --------------------------- Multi-scale ---------------------------- #

        '''
        # out1
        A1 = self.Atrous_r1(self.Atrous_b1(self.Atrous_c1(out1)))
        A3 = self.Atrous_r3(self.Atrous_b3(self.Atrous_c3(out1)))
        A5 = self.Atrous_r5(self.Atrous_b5(self.Atrous_c5(out1)))
        A7 = self.Atrous_r7(self.Atrous_b7(self.Atrous_c7(out1)))
        out1 = torch.cat([out1,A1, A3, A5, A7], dim=1)
        out1 = self.Aconv(out1)
        '''

        # out2
        A1 = self.Atrous_r1_2(self.Atrous_b1_2(self.Atrous_c1_2(out2)))
        A3 = self.Atrous_r3_2(self.Atrous_b3_2(self.Atrous_c3_2(out2)))
        A5 = self.Atrous_r5_2(self.Atrous_b5_2(self.Atrous_c5_2(out2)))
        A7 = self.Atrous_r7_2(self.Atrous_b7_2(self.Atrous_c7_2(out2)))
        out2 = torch.cat([out2, A1, A3, A5, A7], dim=1)
        out2 = self.Aconv_2(out2)
        # out3
        A1 = self.Atrous_r1_3(self.Atrous_b1_3(self.Atrous_c1_3(out3)))
        A3 = self.Atrous_r3_3(self.Atrous_b3_3(self.Atrous_c3_3(out3)))
        A5 = self.Atrous_r5_3(self.Atrous_b5_3(self.Atrous_c5_3(out3)))
        A7 = self.Atrous_r7_3(self.Atrous_b7_3(self.Atrous_c7_3(out3)))
        out3 = torch.cat([out3, A1, A3, A5, A7], dim=1)
        out3 = self.Aconv_3(out3)
        # out4
        A1 = self.Atrous_r1_4(self.Atrous_b1_4(self.Atrous_c1_4(out4)))
        A3 = self.Atrous_r3_4(self.Atrous_b3_4(self.Atrous_c3_4(out4)))
        A5 = self.Atrous_r5_4(self.Atrous_b5_4(self.Atrous_c5_4(out4)))
        A7 = self.Atrous_r7_4(self.Atrous_b7_4(self.Atrous_c7_4(out4)))
        out4 = torch.cat([out4, A1, A3, A5, A7], dim=1)
        out4 = self.Aconv_4(out4)
        # out5
        A1 = self.Atrous_r1_5(self.Atrous_b1_5(self.Atrous_c1_5(out5)))
        A3 = self.Atrous_r3_5(self.Atrous_b3_5(self.Atrous_c3_5(out5)))
        A5 = self.Atrous_r5_5(self.Atrous_b5_5(self.Atrous_c5_5(out5)))
        A7 = self.Atrous_r7_5(self.Atrous_b7_5(self.Atrous_c7_5(out5)))
        out5 = torch.cat([out5, A1, A3, A5, A7], dim=1)
        out5 = self.Aconv_5(out5)

        # ------------------------ Attentive ConvLSTM 2 ------------------------- #
        new_h = out5
        # cell 1
        out5_ori = out5
        f5 = self.conv_fcn2_5(out5)
        h_c = self.conv_h_5(new_h)

        fh5 = f5 + h_c
        fh5 = self.pool_avg_5(fh5)
        fh5 = self.conv_c_5(fh5)
        # Scene Context weighted Module
        w5 = torch.mul(F.softmax(fh5, dim=1), 64)
        fw5 = torch.mul(w5, out5_ori)

        combined = torch.cat((fw5, new_h), dim=1)
        A = self.conv_cell(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        cell_out1 = self.pred_cell_out1(new_h)

        # cell 2
        out4_ori = out4 +out5
        f4 = self.conv_fcn2_4(out4_ori)
        h_c = self.conv_h_4(new_h)

        fh4 = f4 + h_c
        fh4 = self.pool_avg_4(fh4)
        fh4 = self.conv_c_4(fh4)
        # Scene Context weighted Module
        w4 = torch.mul(F.softmax(fh4, dim=1), 64)
        fw4 = torch.mul(w4, out4_ori)

        combined = torch.cat((fw4, new_h), dim=1)
        A = self.conv_cell(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        cell_out2 = self.pred_cell_out2(new_h)

        # cell 3
        out3_ori = out4 + out5 + out3
        f3 = self.conv_fcn2_3(out3_ori)
        h_c = self.conv_h_3(new_h)

        fh3 = f3 + h_c
        fh3 = self.pool_avg_3(fh3)
        fh3 = self.conv_c_3(fh3)
        # Scene Context weighted Module
        w3 = torch.mul(F.softmax(fh3, dim=1), 64)
        fw3 = torch.mul(w3, out3_ori)

        combined = torch.cat((fw3, new_h), dim=1)
        A = self.conv_cell(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        cell_out3 = self.pred_cell_out3(new_h)

        # cell 2
        out2_ori = out2 + out4 + out5 + out3
        f2 = self.conv_fcn2_2(out2_ori)
        h_c = self.conv_h_2(new_h)

        fh2 = f2 + h_c
        fh2 = self.pool_avg_2(fh2)
        fh2 = self.conv_c_2(fh2)
        # Scene Context weighted Module
        w2 = torch.mul(F.softmax(fh2, dim=1), 64)
        fw2 = torch.mul(w2, out2_ori)

        combined = torch.cat((fw2, new_h), dim=1)
        A = self.conv_cell(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)

        output = new_h

        # --------------------- Middle layer supervision ------------------- #
        # pred2
        output2 = self.prediction2(out2)
        outputs2 = F.upsample(output2, scale_factor=4, mode='bilinear')

        # pred3
        output3 = self.prediction3(out3)
        outputs3 = F.upsample(output3, scale_factor=4, mode='bilinear')

        # pred4
        output4 = self.prediction4(out4)
        outputs4 = F.upsample(output4, scale_factor=4, mode='bilinear')

        # pred5
        output5 = self.prediction5(out5)
        outputs5 = F.upsample(output5, scale_factor=4, mode='bilinear')


        # -------------------- ConvLSTM cell_out supervision -----------------------#
        cell_out1 = F.upsample(cell_out1, scale_factor=4, mode='bilinear')
        cell_out2 = F.upsample(cell_out2, scale_factor=4, mode='bilinear')
        cell_out3 = F.upsample(cell_out3, scale_factor=4, mode='bilinear')


        # -------------------------- prediction ----------------------------#
        # final
        output = self.prediction(output)
        outputs_o = F.upsample(output, scale_factor=4, mode='bilinear')


        return outputs_o, outputs2, outputs3, outputs4, outputs5, cell_out1, cell_out2, cell_out3

def ConvLSTM_net():
    model = ConvLSTM(n_class=1)
    return model

if __name__ == "__main__":

    with torch.no_grad():

        net = ConvLSTM_net()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = net.to(device)

        summary(model,(13,64,64,64))