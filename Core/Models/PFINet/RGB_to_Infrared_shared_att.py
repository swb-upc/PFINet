import  torch.nn as nn
from PIL import Image
from torch.nn import init
import numpy as np

from Core.Models.backbones.resnet import resnet50
from Core.loss import Loss
from Core.Models.backbones.resnet_ibn_a import resnet50_ibn_a
import torch

from configs.default.Rgb_to_Infrared_strategy import strategy_cfg
from tools.metrics import *
from Core.attentions.attentions import *
from configs.default.dataset import dataset_cfg
# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


class RGB_to_Infrared(nn.Module):
    def __init__(self, cfg,feature_dim = 5888,num_class=395):
        super(RGB_to_Infrared, self).__init__()
        self.cfg = cfg
        self.feature_dim = feature_dim
        self.num_class = num_class
        #backbone
        self.backbone = resnet50_ibn_a(last_stride=1,pretrained=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        #space attention
        self.space_att = nn.Conv2d(in_channels=2048,out_channels=1,kernel_size=1)

        self.activation = nn.Sigmoid()


        #horizontal
        self.horizontal1 = nn.Conv2d(256,256,kernel_size=1)
        self.horizontal2 = nn.Conv2d(512,512,kernel_size=1)
        self.horizontal3 = nn.Conv2d(1024,1024,kernel_size=1)

        self.up_sample2 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(512,256,kernel_size=1)
        )
        self.up_sample3 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(1024,512,kernel_size=1)
        )
        self.up_sample4 = nn.Sequential(
            nn.Conv2d(2048,1024,kernel_size=1)
        )

        #Conv_unit1'
        self.Conv_unit1 = nn.Conv2d(256, 256, kernel_size=3,padding=1)
        self.Conv_unit2 = nn.Conv2d(512, 512, kernel_size=3,padding=1)
        self.Conv_unit3 = nn.Conv2d(1024, 1024, kernel_size=3,padding=1)



        #bottleneck
        self.bottleneck = nn.BatchNorm1d(num_features=self.feature_dim)
        self.bottleneck.bias.requires_grad_(False)

        #classifier
        self.classifier = nn.Linear(self.feature_dim,self.num_class,bias=False)

        #loss
        self.loss = Loss(self.cfg)

        #weight of RGB-to-Gray
        if self.cfg.Learning_W_RGB:
            self.w_RGB = nn.Parameter(torch.randn(size=(1,3,1,1)))
    def forward(self, x, ids,cam_ids):

        sub = (cam_ids == 3) + (cam_ids == 6)
        if self.cfg.Learning_W_RGB:
            x[sub==0] *= self.w_RGB

        feat = self.backbone.conv1(x)
        feat = self.backbone.bn1(feat)
        feat = self.backbone.relu(feat)
        feat = self.backbone.maxpool(feat)
        feat1 = self.backbone.layer1(feat)
        feat2 = self.backbone.layer2(feat1)
        feat3 = self.backbone.layer3(feat2)
        feat4 = self.backbone.layer4(feat3)

        #horizontal
        feat1_horizon = self.horizontal1(feat1)
        feat2_horizon = self.horizontal2(feat2)
        feat3_horizon = self.horizontal3(feat3)

        #FPN  top-bottom
        up_feat4 = self.up_sample4(feat4)
        up_feat4 = up_feat4 + feat3_horizon
        up_feat3 = self.up_sample3(up_feat4)
        up_feat3 = up_feat3 + feat2_horizon
        up_feat2 = self.up_sample2(up_feat3)
        up_feat2 = up_feat2 + feat1_horizon


        feat_final1 = self.Conv_unit1(up_feat2)
        feat_final2 = self.Conv_unit2(up_feat3)
        feat_final3 = self.Conv_unit3(up_feat4)

        #space attention
        mask4 = self.activation(self.space_att(feat4))
        mask3 = mask4
        mask2 = F.upsample(mask3,scale_factor=2)
        mask1 = F.upsample(mask2,scale_factor=2)

        feat1_att = mask1 * feat_final1
        feat2_att = mask2 *  feat_final2
        feat3_att = mask3 * feat_final3
        feat4_att = mask4 * feat4


        feat1_att = self.avg_pool(feat1_att).view(up_feat2.size(0),-1)
        feat2_att = self.avg_pool(feat2_att).view(up_feat3.size(0), -1)
        feat3_att = self.avg_pool(feat3_att).view(up_feat4.size(0), -1)
        feat4_att = self.avg_pool(feat4_att).view(feat4_att.size(0), -1)
        feat4_ = self.avg_pool(feat4).view(feat4.size(0), -1)

        feat_pool = torch.cat([feat4_,feat4_att,feat3_att,feat2_att,feat1_att],dim=-1)
        #feat_pool = torch.cat([feat4_att,feat3_att,feat2_att,feat1_att],dim=-1)

        bn_feat = self.bottleneck(feat_pool)
        print(bn_feat[0,4096:,])
        cls = self.classifier(bn_feat)
        if self.training:
            loss = self.loss(feat_pool,ids,cls,cam_ids)

            return loss,feat_pool
        else:
            return bn_feat

if __name__ == '__main__':
    input = torch.randn(size=(128,3,384,128))
    ids = torch.randint(high=395,size=(128,))
    camids = ids
    cfg = strategy_cfg
    cfg.merge_from_file('D:\??????\MyCode\Cross_Modality_ReID\configs\RGB_to_Infrared\cfg.yml')


    dataset_cfg = dataset_cfg.get(cfg.dataset)
    for k,v in dataset_cfg.items():
        cfg[k] = v
    cfg.batch_size = cfg.p_size * cfg.k_size

    cfg.freeze()
    model = RGB_to_Infrared(cfg)
    model.eval()
    out = model(input,ids,camids)
    print(out.size())