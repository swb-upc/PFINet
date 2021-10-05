import torch
import torch.nn as nn
from .metric import *
import torch.nn.functional as F

class RankingLoss(nn.Module):

    def __init__(self):
        super(RankingLoss, self).__init__()


    def _label2similarity(sekf, label1, label2):
        '''
        compute similarity matrix of label1 and label2
        :param label1: torch.Tensor, [m]
        :param label2: torch.Tensor, [n]
        :return: torch.Tensor, [m, n], {0, 1}
        '''
        m, n = len(label1), len(label2)
        l1 = label1.view(m, 1).expand([m, n])
        l2 = label2.view(n, 1).expand([n, m]).t()
        similarity = l1 == l2
        return similarity

    def _batch_hard(self, mat_distance, mat_similarity, more_similar):

        if more_similar is 'smaller':
            sorted_mat_distance, index1 = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1,
                                                     descending=True)
            hard_p = sorted_mat_distance[:, 0]
            sorted_mat_distance, index2 = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1,
                                                     descending=False)
            hard_n = sorted_mat_distance[:, 0]
            return hard_p, hard_n, index1[:, 0], index2[:0]

        elif more_similar is 'larger':
            sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (1 - mat_similarity), dim=1,
                                                descending=False)
            hard_p = sorted_mat_distance[:, 0]
            sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (mat_similarity), dim=1, descending=True)
            hard_n = sorted_mat_distance[:, 0]
            return hard_p, hard_n


class TripletLoss(RankingLoss):
    '''
    Compute Triplet loss augmented with Batch Hard
    Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
    '''

    def __init__(self, margin, metric = 'euclidean'):
        '''
        :param margin: float or 'soft', for MarginRankingLoss with margin and soft margin
        :param bh: batch hard
        :param metric: l2 distance or cosine distance
        '''
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.margin_loss = nn.MarginRankingLoss(margin=margin)
        self.metric = metric

    def forward(self, emb1, emb2, emb3, label1, label2, label3):
        '''

        :param emb1: torch.Tensor, [m, dim]
        :param emb2: torch.Tensor, [n, dim]
        :param label1: torch.Tensor, [m]
        :param label2: torch.Tensor, [b]
        :return:
        '''

        if self.metric == 'cosine':
            mat_dist = cosine_dist(emb1, emb2)
            mat_sim = self._label2similarity(label1, label2)
            hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

            mat_dist = cosine_dist(emb1, emb3)
            mat_sim = self._label2similarity(label1, label3)
            _, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

            margin_label = -torch.ones_like(hard_p)

        elif self.metric == 'euclidean':
            mat_dist = euclidean_dist(emb1, emb2)
            mat_sim = self._label2similarity(label1, label2)
            hard_p, _, index1, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')

            mat_dist = euclidean_dist(emb1, emb3)
            mat_sim = self._label2similarity(label1, label3)
            _, hard_n, _, index2 = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')

            margin_label = torch.ones_like(hard_p)
        # 计算相似性损失

        return self.margin_loss(hard_n, hard_p, margin_label)


class SimilarLoss:
    def __init__(self):
        pass

    def cal_loss(self, x, y, label1, label2, mini=1e-8):
        for (i, x_item), (j, y_item) in enumerate(x), enumerate(y):
            x = F.softmax(x, dim=-1)
            y = F.softmax(y, dim=-1)
            loss = torch.sum(x * torch.log(mini + x / (y + mini)), dim=-1) + \
                   torch.sum(y * torch.log(mini + y / (x + mini)), dim=-1)

        else:
            loss = torch.sum(1 / (torch.abs(x - y) + mini), dim=-1)