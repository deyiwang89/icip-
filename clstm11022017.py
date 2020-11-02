import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

#####-------------se layer-------------#####
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()
#####-------------CFM-------------#####
class CFM(nn.Module):
    def __init__(self):
        super(CFM, self).__init__()
        self.conv1h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1h   = nn.BatchNorm2d(64)
        self.conv2h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2h   = nn.BatchNorm2d(64)
        self.conv3h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3h   = nn.BatchNorm2d(64)
        self.conv4h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4h   = nn.BatchNorm2d(64)

        self.conv1v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1v   = nn.BatchNorm2d(64)
        self.conv2v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2v   = nn.BatchNorm2d(64)
        self.conv3v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3v   = nn.BatchNorm2d(64)
        self.conv4v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4v   = nn.BatchNorm2d(64)

    def forward(self, left, down):
        if down.size()[2:] != left.size()[2:]:
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear')
        out1h = F.relu(self.bn1h(self.conv1h(left )), inplace=True)
        out2h = F.relu(self.bn2h(self.conv2h(out1h)), inplace=True)
        out1v = F.relu(self.bn1v(self.conv1v(down )), inplace=True)
        out2v = F.relu(self.bn2v(self.conv2v(out1v)), inplace=True)
        fuse  = out2h*out2v
        out3h = F.relu(self.bn3h(self.conv3h(fuse )), inplace=True)+out1h
        out4h = F.relu(self.bn4h(self.conv4h(out3h)), inplace=True)
        out3v = F.relu(self.bn3v(self.conv3v(fuse )), inplace=True)+out1v
        out4v = F.relu(self.bn4v(self.conv4v(out3v)), inplace=True)
        return out4h, out4v

    def initialize(self):
        weight_init(self)

#####-------------CFD-------------#####
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.cfm45  = CFM()
        self.cfm34  = CFM()
        self.cfm23  = CFM()

    def forward(self, out2h, out3h, out4h, out5v, fback=None):
        if fback is not None:
            refine5      = F.interpolate(fback, size=out5v.size()[2:], mode='bilinear')
            refine4      = F.interpolate(fback, size=out4h.size()[2:], mode='bilinear')
            refine3      = F.interpolate(fback, size=out3h.size()[2:], mode='bilinear')
            refine2      = F.interpolate(fback, size=out2h.size()[2:], mode='bilinear')
            out5v        = out5v+refine5
            out4h, out4v = self.cfm45(out4h+refine4, out5v)
            out3h, out3v = self.cfm34(out3h+refine3, out4v)
            out2h, pred  = self.cfm23(out2h+refine2, out3v)
        else:
            out4h, out4v = self.cfm45(out4h, out5v)
            out3h, out3v = self.cfm34(out3h, out4v)
            out2h, pred  = self.cfm23(out2h, out3v)
        return out2h, out3h, out4h, out5v, pred

    def initialize(self):
        weight_init(self)


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()
# -------------------added by xx------------------- #
def conv1x1(in_planes, out_planes, stride =1):
    '''1*1 convolution'''
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

####################################  Memory-oriented Decoder  #####################################
class ConvLSTM(nn.Module):
    def __init__(self,n_class=2):
        super(ConvLSTM, self).__init__()

        # ------------------- CFM -------------------- #
        self.cfm  = CFM()
        self.conv1_1 = conv1x1(13, 1)
        self.conv_squeeze = conv1x1(128, 64)
        self.out_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.out_relu = nn.ReLU(inplace=True)
         # attentive convlstm 2
        self.conv_cell = nn.Conv2d(64 + 64, 4 * 64, 5, padding=2)

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




        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def convlstm_cell(self,A, new_c):
        (ai, af, ao, ag) = torch.split(A, A.size()[1] // 4, dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ag)
        g = torch.tanh(ag)
        new_c = f * new_c + i * g
        new_h = o * torch.tanh(new_c)
        return new_c , new_h




    def forward(self, level1, level2, level3, level4, level5):

        lb, lc, lw, lh = level5.size()
        out2h,out2v = self.cfm(level2, level1)
        out2_op = torch.cat((out2h,out2v),1).view(lb,lc*2,lw,lh)
        out2_op = self.out_relu(self.out_bn((self.conv_squeeze(out2_op)).view(lb,lc,lw,lh)))

        out3h,out3v = self.cfm(level3, out2_op)
        out3_op = torch.cat((out3h,out3v),1).view(lb,lc*2,lw,lh)
        out3_op = self.out_relu(self.out_bn((self.conv_squeeze(out3_op)).view(lb,lc,lw,lh)))

        out4h,out4v = self.cfm(level4, out3_op)
        out4_op = torch.cat((out4h,out4v),1).view(lb,lc*2,lw,lh)
        out4_op = self.out_relu(self.out_bn((self.conv_squeeze(out4_op)).view(lb,lc,lw,lh)))

        out5h,out5v = self.cfm(level5, out4_op)
        out5_op = torch.cat((out5h,out5v),1).view(lb,lc*2,lw,lh)
        out5_op = self.out_relu(self.out_bn((self.conv_squeeze(out5_op)).view(lb,lc,lw,lh)))

        # out1 = (self.conv1x1(out2_op.transpose(0,1))).transpose(0,1)
        out2 = self.out_relu(self.out_bn((self.conv1_1(out2_op.transpose(0,1))).transpose(0,1)))
        out3 = self.out_relu(self.out_bn((self.conv1_1(out3_op.transpose(0,1))).transpose(0,1)))
        out4 = self.out_relu(self.out_bn((self.conv1_1(out4_op.transpose(0,1))).transpose(0,1)))
        out5 = self.out_relu(self.out_bn((self.conv1_1(out5_op.transpose(0,1))).transpose(0,1)))

        new_c = self.out_relu(self.out_bn((self.conv1_1(out5h.transpose(0,1))).transpose(0,1)))
 
        # # --------------------------- Multi-scale ---------------------------- #


        # # # out1
        # # A1 = self.Atrous_r1(self.Atrous_b1(self.Atrous_c1(out1)))
        # # A3 = self.Atrous_r3(self.Atrous_b3(self.Atrous_c3(out1)))
        # # A5 = self.Atrous_r5(self.Atrous_b5(self.Atrous_c5(out1)))
        # # A7 = self.Atrous_r7(self.Atrous_b7(self.Atrous_c7(out1)))
        # # out1 = torch.cat([out1,A1, A3, A5, A7], dim=1)
        # # out1 = self.Aconv(out1)
    

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



