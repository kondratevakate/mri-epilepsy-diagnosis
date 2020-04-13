import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _triple


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ConvBlock(nn.Module):
    """
    (BN => ReLU => conv )
    """
    def __init__(self, in_channels, out_channels, kernel, stride, padding=1, bayes = False):
        super(ConvBlock, self).__init__()
        if bayes:
            self.conv = nn.Sequential(
                nn.InstanceNorm3d(in_channels),
                nn.ReLU(inplace=True),
                BayesConv3d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=False))
        else:
            self.conv = nn.Sequential(
                nn.InstanceNorm3d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=False))

    def forward(self, x):
        x = self.conv(x)
        return x


class BasicDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, bayes=False):
        super(BasicDownBlock, self).__init__()
        if downsample:
            self.conv_1 = ConvBlock(in_channels, out_channels, kernel=3, stride=2, bayes=bayes)
        else:
            self.conv_1 = ConvBlock(in_channels, out_channels, kernel=3, stride=1, bayes=bayes)
        self.conv_2 = ConvBlock(out_channels, out_channels, kernel=3, stride=1, bayes=bayes)

        self.down = None
        if downsample:
            self.down = ConvBlock(in_channels, out_channels, kernel=1, stride=2, padding=0, bayes=False)

    def forward(self, inp):
        x = self.conv_1(inp)
        x = self.conv_2(x)
        if self.down is not None:
            return x + self.down(inp)
        else:
            return x + inp


class BasicUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample = True, bayes = False):
        super(BasicUpBlock, self).__init__()

        self.upsample = nn.Sequential(ConvBlock(in_channels, out_channels, kernel=1, stride = 1, padding = 0, bayes = False),
                                     nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
        self.conv_1 = ConvBlock(out_channels, out_channels, kernel=3, stride = 1, bayes = bayes)
        self.conv_2 = ConvBlock(out_channels, out_channels, kernel=3, stride = 1, bayes = bayes)


    def forward(self, inp, skip_connection = None):
        x = self.upsample(inp)
        if skip_connection is not None:
            x = x + skip_connection
        x1 = self.conv_1(x)
        x1 = self.conv_2(x1)
        return  x1 + x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


################################################################

class _BayesConvNd(nn.Module):
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_BayesConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.mu_weight = nn.Parameter(torch.Tensor(in_channels, out_channels // groups, *kernel_size))
            self.logsigma_weight = nn.Parameter(torch.Tensor(in_channels, out_channels // groups, *kernel_size))
        else:
            self.mu_weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
            self.logsigma_weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.mu_bias = nn.Parameter(torch.Tensor(out_channels))
            self.logsigma_bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('logsigma_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        self.mu_weight.data.normal_(0, 0.02)
        self.logsigma_weight.data.fill_(-5)
        # nn.init.xavier_uniform_(self.mu_weight)
        # nn.init.xavier_uniform_(self.logsigma_weight, gain=1e-2)
        if self.mu_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.mu_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.mu_bias, -bound, bound)

            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.logsigma_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.logsigma_bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.mu_bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class BayesConv2d(_BayesConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, zero_mean=False, threshold=3):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.zero_mean = zero_mean
        self.threshold = threshold

        super(BayesConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                          padding, dilation, False, _pair(0), groups, bias)
        if zero_mean:
            self.mu_weight = nn.Parameter(torch.zeros_like(self.mu_weight))

    def forward(self, input):
        self.log_alpha = torch.clamp(
            self.logsigma_weight - torch.log(self.mu_weight ** 2 + 1e-8), -5, 5)
        if self.logsigma_bias is not None:
            bias = torch.pow(self.logsigma_bias, 2)
        else:
            bias = None

        if self.training:
            sigma_out = torch.sqrt(1e-4 + F.conv2d(torch.pow(input, 2),
                                                   self.mu_weight ** 2 * torch.exp(
                                                       self.log_alpha), bias,
                                                   self.stride, self.padding,
                                                   self.dilation, self.groups))
            mu_out = F.conv2d(input, self.mu_weight, self.mu_bias, self.stride,
                              self.padding, self.dilation, self.groups)
        else:
            mask = (self.log_alpha < self.threshold).float()
            mu_out = F.conv2d(input, self.mu_weight * mask, self.mu_bias, self.stride,
                              self.padding,
                              self.dilation, self.groups)
            sigma_out = torch.sqrt(1e-4 + F.conv2d(torch.pow(input, 2),
                                                   self.mu_weight ** 2 * torch.exp(
                                                       self.log_alpha) * mask,
                                                   bias, self.stride, self.padding,
                                                   self.dilation, self.groups))

        eps = sigma_out.data.new(sigma_out.size()).normal_()
        return eps.mul(sigma_out) + mu_out


class BayesConv3d(_BayesConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, zero_mean=False, threshold=3):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        self.zero_mean = zero_mean
        self.threshold = threshold

        super(BayesConv3d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                          padding, dilation, False, _triple(0), groups, bias)
        if zero_mean:
            self.mu_weight = nn.Parameter(torch.zeros_like(self.mu_weight))

    def forward(self, input):
        self.log_alpha = torch.clamp(self.logsigma_weight - torch.log(self.mu_weight**2 + 1e-8), -5, 5)
        if self.logsigma_bias is not None:
            bias = torch.pow(self.logsigma_bias, 2)
        else:
            bias = None

        if self.training:
            sigma_out = torch.sqrt(1e-4 + F.conv3d(torch.pow(input, 2), self.mu_weight**2 * torch.exp(self.log_alpha), bias,
                                                   self.stride, self.padding, self.dilation, self.groups))
            # if self.zero_mean:
            #     mu_out = torch.zeros_like(sigma_out)
            # else:
            mu_out = F.conv3d(input, self.mu_weight, self.mu_bias, self.stride,  self.padding, self.dilation, self.groups)
        else:
            mask = (self.log_alpha < self.threshold).float()
            mu_out = F.conv3d(input, self.mu_weight*mask, self.mu_bias, self.stride,  self.padding,
                              self.dilation, self.groups)
            sigma_out = torch.sqrt(1e-4 + F.conv3d(torch.pow(input, 2), self.mu_weight**2 * torch.exp(self.log_alpha)*mask,
                                                   bias, self.stride, self.padding, self.dilation, self.groups))

        eps = sigma_out.data.new(sigma_out.size()).normal_()
        return eps.mul(sigma_out) + mu_out

#         print('Input 2:', torch.max(torch.clamp(torch.pow(input, 2), min = 1e-5, max = 1e2)).item(),
#               torch.mean(torch.clamp(torch.pow(input, 2), min = 1e-4, max = 1e2)).item(),
#               torch.min(torch.clamp(torch.pow(input, 2), min = 1e-4, max = 1e2)).item())
#         print('Sigma Weight 2:', torch.max(torch.clamp(torch.exp(self.logsigma_weight), min = 1e-4, max = 1e5)).item(),
#               torch.mean(torch.clamp(torch.exp(self.logsigma_weight), min = 1e-4, max = 1e5)).item(),
#               torch.min(torch.clamp(torch.exp(self.logsigma_weight), min = 1e-4, max = 1e5)).item())
#         print('Sigma 2:', torch.max(F.conv3d(torch.pow(input, 2), torch.clamp(torch.exp(self.logsigma_weight), 1e-4, 1e4),
#                                                     bias, self.stride, self.padding, self.dilation, self.groups)).item(),
#               torch.mean(F.conv3d(torch.pow(input, 2), torch.clamp(torch.exp(self.logsigma_weight), 1e-4, 1e4),
#                                                     bias, self.stride, self.padding, self.dilation, self.groups)).item(),
#               torch.min(F.conv3d(torch.pow(input, 2), torch.clamp(torch.exp(self.logsigma_weight), 1e-4, 1e4),
#                                                     bias, self.stride, self.padding, self.dilation, self.groups)).item())
#         print('Sigma:', torch.max(sigma_out).item(), torch.mean(sigma_out).item(), torch.min(sigma_out).item())
#         print('Mu:', torch.max(mu_out).item(), torch.mean(mu_out).item())
#         print('Eps:', torch.max(eps).item(), torch.mean(eps).item())
#         print('Output:', torch.max(eps.mul(sigma_out) + mu_out).item(),
        #         torch.mean(eps.mul(sigma_out) + mu_out).item(), torch.sum(eps.mul(sigma_out) + mu_out).item())
#         print()



########################################################################

class ConvSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvSample, self).__init__()
        self.conv_mu = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv_sigma = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        mu =  self.conv_mu(x)
        logsigma = self.conv_sigma(torch.log(x.pow(2)+1e-8))

        std = torch.exp(0.5*logsigma)
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std) + mu



class DeFlatten(nn.Module):
    def __init__(self,shape):
        super(DeFlatten, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(input.size(0), self.shape[0], self.shape[1], self.shape[2], self.shape[3])


class Conv_Layer(nn.Module):
    '''(conv => BN => ReLU)'''
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv_Layer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride = stride),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x



class Conv_Transpose_Layer(nn.Module):
    '''(conv transpose => BN => ReLU)'''
    def __init__(self, in_channels, out_channels, stride=2, kernel_size = (4, 4, 4)):
        super(Conv_Transpose_Layer, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels,
                               kernel_size = kernel_size, stride=stride),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class Down_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down_Conv, self).__init__()
        self.conv_1 = Conv_Layer(in_channels, out_channels, 2)
        self.conv_2 = Conv_Layer(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv_2(self.conv_1(x))
        return x

class Init_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Init_Conv, self).__init__()
        self.conv_1 = Conv_Layer(in_channels, out_channels, stride = 1)
        self.conv_2 = Conv_Layer(out_channels, out_channels, stride = 1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x


class Up_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up_Conv, self).__init__()
        self.deconv = Conv_Transpose_Layer(in_channels, in_channels//2)
        self.conv = Conv_Layer(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.deconv(x1)

        diffZ = x2.size()[4] - x1.size()[4]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, (diffZ // 2, diffZ - diffZ//2,
                        diffY // 2, diffY - diffY//2,
                        diffX // 2, diffX - diffX//2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Final_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Final_Conv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 1)
#         self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
#         x = self.relu(x)
        return x
