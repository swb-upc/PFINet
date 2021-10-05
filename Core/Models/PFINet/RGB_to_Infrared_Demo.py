import  torch.nn as nn
from PIL import Image
from torch.nn import init
import numpy as np

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

class gelu(nn.Module):
    def __init__(self):
        super(gelu, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+F.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))


class SpaceAttention(nn.Module):
    def __init__(self,inchannels):
        super(SpaceAttention, self).__init__()
        self.att = nn.Sequential(
            nn.Conv2d(in_channels=inchannels,out_channels=1,kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self,x):
        mask = self.att(x)
        return x * mask


class RGB_to_Infrared(nn.Module):
    def __init__(self, cfg,feature_dim = 5888,num_class=395):
        super(RGB_to_Infrared, self).__init__()
        self.cfg = cfg
        self.feature_dim = feature_dim
        self.num_class = num_class
        self.mutual_learning = self.cfg.mutual_learning
        print('multual_learning: ',self.mutual_learning)
        #backbone
        self.backbone = resnet50_ibn_a(last_stride=1,pretrained=True)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        #space attention
        self.space_att1 = SpaceAttention(256)
        self.space_att2 = SpaceAttention(512)
        self.space_att3 = SpaceAttention(1024)
        self.space_att4 = SpaceAttention(2048)

        #pool
        self.avg_pool1 = nn.Sequential(
            #nn.AdaptiveMaxPool2d((32,32)),
            nn.AdaptiveAvgPool2d(1)
        )
        self.avg_pool2 = nn.Sequential(
            #nn.AdaptiveMaxPool2d((16,16)),
            nn.AdaptiveAvgPool2d(1)
        )
        self.avg_pool3 = nn.Sequential(
            #nn.AdaptiveMaxPool2d((8,8)),
            nn.AdaptiveAvgPool2d(1)
        )
        self.avg_pool4 = nn.Sequential(
            #nn.AdaptiveMaxPool2d((4,4)),
            nn.AdaptiveAvgPool2d(1)
        )

        #horizontal
        self.horizontal1 = nn.Conv2d(256,256,kernel_size=1)
        self.horizontal2 = nn.Conv2d(512,512,kernel_size=1)
        self.horizontal3 = nn.Conv2d(1024,1024,kernel_size=1)
        self.horizontal4 = nn.Conv2d(2048,2048,kernel_size=1)

        # self.up_sample1 = nn.Sequential(
        #     # nn.Upsample(scale_factor=2, mode='bilinear'),
        #     nn.Conv2d(256, 64, kernel_size=1)
        # )

        self.up_sample2 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(512,256,kernel_size=1)
        )
        self.up_sample3 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(1024,512,kernel_size=1)
        )
        self.up_sample4 = nn.Sequential(
            #nn.Upsample(scale_factor=2,mode='bilinear'),
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
    def forward(self, x):


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
        feat1_att = self.space_att1(feat_final1)
        feat2_att = self.space_att2(feat_final2)
        feat3_att = self.space_att3(feat_final3)
        feat4_att = self.space_att4(feat4)


        feat1 = self.avg_pool1(feat1_att).view(up_feat2.size(0),-1)
        feat2 = self.avg_pool2(feat2_att).view(up_feat3.size(0), -1)
        feat3 = self.avg_pool3(feat3_att).view(up_feat4.size(0), -1)
        feat4 = self.avg_pool4(feat4).view(feat4.size(0), -1)
        feat4_ = self.avg_pool4(feat4_att).view(feat4_att.size(0), -1)
        feat_pool = torch.cat([feat4,feat4_,feat3,feat2,feat1],dim=-1)

        bn_feat = self.bottleneck(feat_pool)
        cls = self.classifier(bn_feat)
        return cls

if __name__ == '__main__':
    input = torch.randn(size=(128,3,384,128))
    ids = torch.randint(high=395,size=(128,))
    camids = ids
    cfg = strategy_cfg
    cfg.merge_from_file('D:\学习\MyCode\Cross_Modality_ReID\configs\RGB_to_Infrared\cfg.yml')


    dataset_cfg = dataset_cfg.get(cfg.dataset)
    for k,v in dataset_cfg.items():
        cfg[k] = v
    cfg.batch_size = cfg.p_size * cfg.k_size

    cfg.freeze()
    model = RGB_to_Infrared(cfg)
    model.eval()
    out = model(input,ids,camids)
    print(out.size())