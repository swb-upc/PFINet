import numpy as np
from torch import nn, tensor
import torch
from torch.autograd import Variable


class center_loss(nn.Module):
    def __init__(self, margin=0.1, dist_type='l2'):
        super(center_loss, self).__init__()
        self.margin = margin
        self.dist_type = dist_type
        if dist_type == 'l2':
            self.dist = nn.MSELoss(reduction='sum')
        if dist_type == 'cos':
            self.dist = nn.CosineSimilarity(dim=0)
        if dist_type == 'l1':
            self.dist = nn.L1Loss()

    def forward(self, feat1, feat2, label1, label2):
        feat_size = feat1.size()[1]
        feat_num = feat1.size()[0]
        label_num = len(label1.unique())
        feat1 = feat1.chunk(label_num, 0)
        feat2 = feat2.chunk(label_num, 0)
        # loss = Variable(.cuda())
        for i in range(label_num):
            center1 = torch.mean(feat1[i], dim=0)
            center2 = torch.mean(feat2[i], dim=0)

            if i == 0:
                dist = torch.pow(feat1[i] - center1, 2).sum(dim=1).sum()
                dist += torch.pow(feat2[i] - center2, 2).sum(dim=1).sum()
            else:
                dist += torch.pow(feat1[i] - center1, 2).sum(dim=1).sum()
                dist += torch.pow(feat2[i] - center2, 2).sum(dim=1).sum()

        return dist

if __name__ == '__main__':
    a = torch.tensor([[0,1,2],[1,2,3],[0,1,2]],dtype=torch.float)
    mean = torch.mean(a,dim=0)
    print('mean',mean)
    b = torch.tensor([1,2,3],dtype=torch.float)
    c = torch.pow(a-b,2).sum(dim=1).sum()
    print(c)