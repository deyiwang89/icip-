import math

from torch.autograd import Variable
import torch.nn.functional as F
import torch

import pytorch_ssim
import pytorch_iou
import torch.nn as nn


running_loss_final = 0
running_loss = 0
running_loss_cell = 0



ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def cross_entropy2d(input, target, weight=None, size_average=True):
    
    n, c, h, w = input.size()
    input_2 = input
    input = input.transpose(1,2).transpose(2,3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    input = input.view(-1, c)

    mask = target >= 0
    target = target[mask]

    bce_out = F.cross_entropy(input, target, weight=weight, size_average=False)
    target = target.view(1, -1, h, w)
    target = torch.cat([target,target],1).float()

    ssim_out = 1 - ssim_loss(input_2,target)
    iou_out = iou_loss(input_2,target)

    loss = bce_out + ssim_out + iou_out

    if size_average:
        loss /= mask.data.sum()
    return loss




class Trainer(object):

    def __init__(self, cuda, model_rgb,
                 model_focal, model_clstm, optimizer_rgb,
                 optimizer_focal,optimizer_clstm,
                 train_loader, max_iter, snapshot, outpath, sshow, size_average=False):
        self.cuda = cuda
        self.model_rgb = model_rgb
        self.model_focal = model_focal
        self.model_clstm = model_clstm
        self.optim_rgb = optimizer_rgb
        self.optim_focal = optimizer_focal
        self.optim_clstm = optimizer_clstm
        self.train_loader = train_loader
        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.snapshot = snapshot
        self.outpath = outpath
        self.sshow = sshow
        self.size_average = size_average



    def train_epoch(self):

        for batch_idx, (data, target, focal) in enumerate(self.train_loader):

            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue # for resuming
            self.iteration = iteration
            if self.iteration >= self.max_iter:
                break
            if self.cuda:
                data, target, focal = data.cuda(), target.cuda(), focal.cuda()
            data, target, focal = Variable(data), Variable(target), Variable(focal)
            basize, dime, height, width = focal.size() # 2*36*256*256
            focal = focal.view(1, basize, dime, height, width).transpose(0, 1) # 2*1*36*256*256

            focal = torch.cat(torch.chunk(focal, 12, dim=2), dim=1) # 2*12*3*256*256
            focal = torch.cat(torch.chunk(focal, basize, dim=0), dim=1).squeeze()   #24* 3x256x256

            self.optim_rgb.zero_grad()
            self.optim_focal.zero_grad()
            self.optim_clstm.zero_grad()

            global running_loss_final
            global running_loss
            global running_loss_cell






            # RGBNet's output
            r1,r2,r3,r4,r5 = self.model_rgb(data)

            # FocalNet's output
            f1,f2,f3,f4,f5 = self.model_focal(focal)  # FocalNet's output

            # concat focal and rgb
            c1 = torch.cat((r1,f1),dim=0)
            c2 = torch.cat((r2, f2), dim=0)
            c3 = torch.cat((r3, f3), dim=0)
            c4 = torch.cat((r4, f4), dim=0)
            c5 = torch.cat((r5, f5), dim=0)

            output, output2, output3, output4, output5, cell1, cell2, cell3 = self.model_clstm(c1,c2,c3,c4,c5)


            # Final Prediction
            loss_focal_clstm1 = cross_entropy2d(output, target, size_average=self.size_average)
            # Aux supervision
            loss_focal_clstm2 = cross_entropy2d(output2, target, size_average=self.size_average)
            loss_focal_clstm3 = cross_entropy2d(output3, target, size_average=self.size_average)
            loss_focal_clstm4 = cross_entropy2d(output4, target, size_average=self.size_average)
            loss_focal_clstm5 = cross_entropy2d(output5, target, size_average=self.size_average)
            loss_focal_clstm = (loss_focal_clstm1+loss_focal_clstm2+loss_focal_clstm3+loss_focal_clstm4+loss_focal_clstm5) / 5
            # cell_out supervision
            loss_cellout1 = cross_entropy2d(cell1, target, size_average=self.size_average)
            loss_cellout2 = cross_entropy2d(cell2, target, size_average=self.size_average)
            loss_cellout3 = cross_entropy2d(cell3, target, size_average=self.size_average)
            loss_cell_out = (loss_cellout1 + loss_cellout2 + loss_cellout3) / 3



            running_loss_final += loss_focal_clstm.item()
            running_loss_cell += loss_cell_out.item()
            running_loss += loss_focal_clstm1.item()


            if iteration % self.sshow == (self.sshow-1):
                # print('\n [%3d, %6d,LFNet loss: %.3f]' % (self.epoch + 1, iteration+1,
                #                                                          running_loss_focal_clstm/self.sshow))
                print('\n [%3d, %6d,   LFRNN loss: %.3f, Cell_out loss: %.3f, Aux loss: %.3f ]' % (self.epoch + 1, iteration +1, running_loss / self.sshow,
                                                                                                   running_loss_cell / self.sshow, running_loss_final / self.sshow))

                running_loss_final = 0.0
                running_loss = 0.0
                running_loss_cell = 0.0



            if iteration <= 0:
                if iteration % self.snapshot == (self.snapshot-1):
                    savename = ('%s/snapshot_iter_%d.pth' % (self.outpath, iteration+1))
                    torch.save(self.model_rgb.state_dict(), savename)
                    print('save: (snapshot: %d)' % (iteration+1))
                
                    savename_focal = ('%s/focal_snapshot_iter_%d.pth' % (self.outpath, iteration+1))
                    torch.save(self.model_focal.state_dict(), savename_focal)
                    print('save: (snapshot_focal: %d)' % (iteration+1))

                    savename_clstm = ('%s/clstm_snapshot_iter_%d.pth' % (self.outpath, iteration+1))
                    torch.save(self.model_clstm.state_dict(), savename_clstm)
                    print('save: (snapshot_clstm: %d)' % (iteration+1))

            else:
                if iteration % 1000 == (1000 - 1):
                    savename = ('%s/snapshot_iter_%d.pth' % (self.outpath, iteration + 1))
                    torch.save(self.model_rgb.state_dict(), savename)
                    print('save: (snapshot: %d)' % (iteration + 1))

                    savename_focal = ('%s/focal_snapshot_iter_%d.pth' % (self.outpath, iteration + 1))
                    torch.save(self.model_focal.state_dict(), savename_focal)
                    print('save: (snapshot_focal: %d)' % (iteration + 1))

                    savename_clstm = ('%s/clstm_snapshot_iter_%d.pth' % (self.outpath, iteration + 1))
                    torch.save(self.model_clstm.state_dict(), savename_clstm)
                    print('save: (snapshot_clstm: %d)' % (iteration + 1))



            if (iteration+1) == self.max_iter:
                savename = ('%s/snapshot_iter_%d.pth' % (self.outpath, iteration+1))
                torch.save(self.model_rgb.state_dict(), savename)
                print('save: (snapshot: %d)' % (iteration+1))

                savename_focal = ('%s/focal_snapshot_iter_%d.pth' % (self.outpath, iteration+1))
                torch.save(self.model_focal.state_dict(), savename_focal)
                print('save: (snapshot_focal: %d)' % (iteration+1))

                savename_clstm = ('%s/clstm_snapshot_iter_%d.pth' % (self.outpath, iteration+1))
                torch.save(self.model_clstm.state_dict(), savename_clstm)
                print('save: (snapshot_clstm: %d)' % (iteration+1))




            loss_focal_clstm.backward(retain_graph=True)
            self.optim_clstm.step()
            self.optim_focal.step()
            self.optim_rgb.step()

            loss_cell_out.backward(retain_graph=True)
            self.optim_clstm.step()
            self.optim_focal.step()
            self.optim_rgb.step()

            loss_focal_clstm1.backward()
            self.optim_clstm.step()
            self.optim_focal.step()
            self.optim_rgb.step()

    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))

        for epoch in range(max_epoch):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
