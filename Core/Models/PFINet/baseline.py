import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn import Parameter
import numpy as np

import cv2

from Core.Models.backbones.resnet import resnet50
from Core.Models.MPANet.utils.calc_acc import calc_acc

from Core.Models.MPANet.layers import TripletLoss
from Core.Models.MPANet.layers import CenterTripletLoss
from Core.Models.MPANet.layers import CenterLoss
from Core.Models.MPANet.layers import cbam
from Core.Models.MPANet.layers import NonLocalBlockND

class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()

        self.backbone = resnet50(pretrained=True, drop_last_stride=True, modality_attention=0)

        self.base_dim = 2048

        self.bn_neck = nn.BatchNorm1d(self.base_dim)
        nn.init.constant_(self.bn_neck.bias, 0) 
        self.bn_neck.bias.requires_grad_(False)

        self.classifier = nn.Linear(self.base_dim , 395, bias=False)

        self.center_cluster_loss = CenterTripletLoss(k_size=8, margin=0.7)

        self.id_loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, inputs, labels=None, **kwargs):
        global_feat = self.backbone(inputs)
        b, c, w, h = global_feat.shape
        feats = F.avg_pool2d(global_feat, global_feat.size()[2:])
        feats = feats.view(feats.size(0), -1)

        if not self.training:
            feats = self.bn_neck(feats)
            return feats
        else:
            loss = 0
            center_cluster_loss, _, _ = self.center_cluster_loss(feats.float(), labels)
            loss+=center_cluster_loss
            feats = self.bn_neck(feats)

            logits = self.classifier(feats)
            cls_loss = self.id_loss(logits.float(), labels)
            loss += cls_loss
            return loss,feats