import torch.nn as nn
import torch
import torch.nn.functional as F

class Self_Space_Attention(nn.Module):
    def __init__(self,inchannel):
        super(Self_Space_Attention, self).__init__()
        self.q = nn.Conv2d(in_channels=inchannel,out_channels= inchannel// 8,kernel_size=1)
        self.k = nn.Conv2d(in_channels=inchannel,out_channels= inchannel // 8,kernel_size=1)
        self.v = nn.Conv2d(in_channels=inchannel,out_channels=inchannel,kernel_size=1)
    def forward(self,x):
        b,c,h,w = x.size()
        Q = self.q(x).view(b,-1,h*w).transpose(1,2)  #b * d * c
        K = self.k(x).view(b,-1,h*w) #b * c * d
        w_q = torch.bmm(Q,K)  #b * d * d
        w_q = F.softmax(w_q,dim=-1)  #b * d * d
        V = self.v(x).view(b,c,-1)  #b * c * d
        V = torch.bmm(V,w_q.permute(0,2,1)).view(b,c,h,w)
        return x + V

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




if __name__ == '__main__':
    a = torch.randn((128,2048,8,4))
    attention = Self_Space_Attention(2048)
    b = attention(a)
    print(b.size())