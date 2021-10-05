# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F

from .triplet_loss import TripletLoss
from .cluster_loss import ClusterLoss
from .center_loss import CenterLoss
from .range_loss import RangeLoss
from .smooth_label_loss import CrossEntropyLabelSmooth
from .heterogeneity_loss import hetero_loss
from .cm_triplet_loss import TripletLoss as CMTripletLoss
from .center_triplet_loss import CenterTripletLoss
from .Circle_Loss import CircleLoss
import torch.nn as nn

__all__ = ['CenterTripletLoss','TripletLoss','ClusterLoss','CenterLoss','RangeLoss','CrossEntropyLabelSmooth','hetero_loss','CMTripletLoss']

class Loss(nn.Module):
    def __init__(self,cfg,**kwargs):
        super(Loss, self).__init__()
        loss_type = []
        self.cfg = cfg
        self.TimeMoving = kwargs.get('TimeMoving')
        if self.cfg.id_loss:
            #self.id_loss = CrossEntropyLabelSmooth(num_classes=cfg.num_id)
            self.id_loss = nn.CrossEntropyLoss()
            loss_type.append('id_loss')
        if self.cfg.Triplet_loss:
            self.Triplet_loss = TripletLoss(margin=cfg.Triplet_margin)
            loss_type.append('Triplet_loss')
        if self.cfg.hetero_loss:
            self.hetero_loss = hetero_loss(margin=0)
            loss_type.append('hetero_loss')
        if self.cfg.CMTriplet_loss:
            self.CMTriplet_loss = CMTripletLoss(margin=cfg.CMTriplet_margin)
            loss_type.append('CMTriplet_loss')
        if self.cfg.center_loss:
            self.center_loss = CenterLoss(num_classes=cfg.num_id, feat_dim=cfg.feature_dim)
            loss_type.append('CenterLoss')
        if self.cfg.CenterTripletLoss:
            self.center_triplet_loss = CenterTripletLoss(k_size=cfg.k_size, margin=0.7)
            loss_type.append('CenterTripletLoss')
        if self.cfg.CircleLoss:
            self.CircleLoss = CircleLoss(0.25,256)

        print('#======',loss_type,'======#')

    def forward(self, x, ids,cls,cam_ids):
        sub = (cam_ids == 3) + (cam_ids == 6)
        loss = 0.0

        if self.cfg.id_loss:
            loss += self.id_loss(cls, ids)

        if self.cfg.Triplet_loss:
            triplet, _, _ = self.Triplet_loss(x, ids)
            loss += triplet


        if self.cfg.CMTriplet_loss:
            loss+=self.CMTriplet_loss(x[sub==1],x[sub==0],x[sub==0],ids[sub==1],ids[sub==0],ids[sub==0])
            loss+=self.CMTriplet_loss(x[sub==0],x[sub==1],x[sub==1],ids[sub==0],ids[sub==1],ids[sub==1])


        if self.cfg.center_loss:
            loss += self.center_loss(x, ids)

        if self.cfg.CenterTripletLoss:
            loss += self.center_triplet_loss(x, ids)[0]

        if self.cfg.hetero_loss:
            sub = (cam_ids == 3) + (cam_ids == 6)
            loss += self.hetero_loss(x[sub==1],x[sub==0],ids[sub==1],ids[sub==0])

        if self.cfg.CircleLoss:
            loss += self.CircleLoss(x,ids)

        return loss