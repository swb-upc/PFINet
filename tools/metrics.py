import torch
import torch.nn as nn

def cosine_dist(x, y):
	'''
	:param x: torch.tensor, 2d
	:param y: torch.tensor, 2d
	:return:
	'''

	bs1 = x.size()[0]
	bs2 = y.size()[0]

	frac_up = torch.matmul(x, y.transpose(0, 1))
	frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
	            (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
	cosine = frac_up / frac_down

	return cosine


def euclidean_dist(x, y):
	"""
	Args:
	  x: pytorch Variable, with shape [m, d]
	  y: pytorch Variable, with shape [n, d]
	Returns:
	  dist: pytorch Variable, with shape [m, n]
	"""
	m, n = x.size(0), y.size(0)
	xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
	yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
	dist = xx + yy
	dist.addmm_(1, -2, x, y.t())
	dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
	return dist

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

if __name__ == '__main__':
	a = torch.tensor([[1, 1], [2, 2], [3, 3]]).float()
	b = torch.tensor([[1, 1], [2, 2], [3, 3]]).float()
	c = euclidean_dist(a,b)
	print(c)
	x,y = torch.sort(c,dim=-1)
	print(x[:,0])
	print(y[:,0])