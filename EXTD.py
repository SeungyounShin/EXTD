#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from layers import *
from data.config import cfg
import numpy as np

class feature_extractor(nn.Module):
    def __init__(self,C_out):
        super(feature_extractor, self).__init__()
        self.conv = nn.Conv2d(3,C_out,3,2,1)
        self.BatchNorm = nn.BatchNorm2d(C_out)
        self.activation = nn.ReLU()
    def forward(self,x):
        x = self.conv(x)
        x = self.BatchNorm(x)
        return self.activation(x)

class inverted_residual_1(nn.Module):
    def __init__(self,C_in,C_out):
        super(inverted_residual_1, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.conv1 = nn.Conv2d(C_in,C_in,3,1,padding=1,groups=C_in)
        self.conv2 = nn.Conv2d(C_in,C_out,1,1,padding=0,groups=1)
        self.batchNorm1 = nn.BatchNorm2d(C_in)
        self.batchNorm2 = nn.BatchNorm2d(C_out)
        self.activation = nn.PReLU()
    def forward(self,x):
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        return self.batchNorm2(x)

class inverted_residual_2(nn.Module):
    def __init__(self,C_in,h,stride=2):
        super(inverted_residual_2, self).__init__()
        self.conv1 = nn.Conv2d(C_in,h,1,1,padding=0,groups=1)
        self.conv2 = nn.Conv2d(h,h,3,stride,padding=1,groups=h)
        self.conv3 = nn.Conv2d(h,C_in,1,1,padding=0,groups=1)
        self.batchNorm1 = nn.BatchNorm2d(h)
        self.batchNorm2 = nn.BatchNorm2d(h)
        self.batchNorm3 = nn.BatchNorm2d(C_in)
        self.activation = nn.PReLU()
    def forward(self,x):
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.activation(x)
        x = self.conv3(x)
        return self.batchNorm3(x)

class upsample(nn.Module):
    def __init__(self, C_in, filters = 64):
        super(upsample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear').cuda()
        self.depthwise_conv = nn.Conv2d(C_in,filters,3,1,1,groups=C_in).cuda()
        self.pointwise_conv = nn.Conv2d(filters,C_in,1,1,0,groups=1).cuda()
        self.batchNorm = nn.BatchNorm2d(C_in).cuda()
        self.activation = nn.ReLU().cuda()
    def forward(self,x):
        x = self.upsample(x)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.batchNorm(x)
        return self.activation(x)

class backbone(nn.Module):
    def __init__(self,filters = 64):
        super(backbone, self).__init__()
        self.blocks = []
        self.blocks.append(inverted_residual_1(filters,filters).cuda())
        for i in range(3):
            self.blocks.append(inverted_residual_2(filters,filters*2,1).cuda())
        self.blocks.append(inverted_residual_2(filters,filters*2,2).cuda())
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        # print(x)
        return x

def class_predictor(C_in,filters=2):
    return nn.Conv2d(C_in,filters,3,1,1)

def bbox_predictor(C_in):
    return nn.Conv2d(C_in,4,3,1,1)

class EXTD(nn.Module):
    def __init__(self,phase):
        super(EXTD, self).__init__()
        self.phase = phase
        self.num_classes = 2
        self.E = feature_extractor(64)
        self.F = backbone()
        self.up, self.cls_pred, self.bbox_pred = [],[],[]
        for i in range(5):
            self.up.append(upsample(64))
        for i in range(6):
            if(i==0):
                #maxout
                self.cls_pred.append(class_predictor(64,4).cuda())
            else:
                self.cls_pred.append(class_predictor(64).cuda())
            self.bbox_pred.append(bbox_predictor(64).cuda())

    def forward(self,x):
        size = x.size()[2:]
        x = self.E(x)
        f,g = [],[]

        #SSD
        for i in range(6):
            x = self.F(x)
            f.append(x)

        #FPN
        g.append(self.up[0](f[-1])+f[-2])
        for i in range(4):
            g.append(self.up[i+1](g[-1])+f[-3-i])


        #head
        conf, loc, feature_maps = [],[], [f[-1]]+g
        for i in range(6):
            feature_map = feature_maps[i]

            if(i==0):
                #maxout
                conf_x = self.cls_pred[i](feature_map).permute(0, 2, 3, 1).contiguous()
                max_conf, _ = torch.max(conf_x[:,:, :, 0:3], dim=3, keepdim=True)
                conf_x = torch.cat((max_conf, conf_x[:, :, :, 3:]), dim=3)
                conf.append(conf_x)

            else:
                conf.append(self.cls_pred[i](feature_map).permute(0, 2, 3, 1).contiguous())
            loc.append(self.bbox_pred[i](feature_map).permute(0, 2, 3, 1).contiguous())

        features_maps_size = []
        for i in range(len(loc)):
            feat = []
            feat += [loc[i].size(1), loc[i].size(2)]
            features_maps_size += [feat]


        self.priorbox = PriorBox(size, features_maps_size,cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        #print(self.priors[0], self.priors[-1])
        if self.phase == 'test':
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,self.num_classes)),      # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )

        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )

        return output

    def xavier(self, param):
        init.xavier_uniform(param)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            self.xavier(m.weight.data)
            m.bias.data.zero_()

def build_extd(phase, num_classes=2):
    return EXTD(phase)
