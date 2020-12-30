import torch
from torch import nn as nn


class UNet(nn.Module):
    def __init__(self, num_filter, num_class):
        super(UNet, self).__init__()
        kernel = 3
        padding = kernel//2
        self.downconv1 = seqLayersDown(1, num_filter, kernel, 1)
        self.downconv2 = seqLayersDown(num_filter, num_filter*2, kernel, padding)
        self.downconv3 = seqLayersDown(num_filter*2, num_filter*4, kernel, padding)
        self.downconv4 = seqLayersDown(num_filter*4, num_filter*8, kernel, padding)

        self.rfconv = nn.Sequential(
                        nn.Conv2d(num_filter*8, num_filter*8, kernel_size=kernel, padding=padding),
                        nn.BatchNorm2d(num_filter*8),
                        nn.ReLU()
                     )

        self.upconv4 = seqLayersUp(num_filter*8 + num_filter*8, num_filter*8, kernel, padding)
        self.upconv3 = seqLayersUp(num_filter*8 + num_filter*4, num_filter*4, kernel, padding)
        self.upconv2 = seqLayersUp(num_filter*4 + num_filter*2, num_filter*2, kernel, padding)
        self.upconv1 = seqLayersUp(num_filter*2 + num_filter, 3, kernel, padding)

        self.finalconv = nn.Conv2d(1+3, num_class,  kernel_size=kernel, padding=padding)


    def forward(self, x):
        self.out1 = self.downconv1(x)
        self.out2 = self.downconv2(self.out1)
        self.out3 = self.downconv3(self.out2)
        self.out4 = self.downconv4(self.out3)
        self.rfOut = self.rfconv(self.out4)
        self.out5 = self.upconv4(torch.cat((self.rfOut, self.out4), dim = 1))
        self.out6 = self.upconv3(torch.cat((self.out5, self.out3), dim = 1))
        self.out7 = self.upconv2(torch.cat((self.out6, self.out2), dim = 1))
        self.out8 = self.upconv1(torch.cat((self.out7, self.out1), dim = 1))
        self.out_final = self.finalconv(torch.cat((self.out8, x), dim=1))
        return self.out_final


def seqLayersUp(in_channel, out_channel, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel, padding=padding),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
        nn.Upsample(scale_factor=2),
    )


def seqLayersDown(in_channel, out_channel, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel, padding=padding),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predict, target):
        predict = predict.contiguous()
        target = target.contiguous()
        intersect = (predict * target).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersect + 1.) / (predict.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + 1.)))
        return loss.mean()