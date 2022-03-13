import torch
from torch import nn

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
cls = nn.CrossEntropyLoss()


def infonce_loss(anchor, positive, negative, alpha=0.4, device='cuda'):
    pos_dist = -cos(anchor, positive)
    anchor = anchor.repeat(10, 1)
    neg_dist = -cos(anchor, negative)
    # print(pos_dist, neg_dist)
    dis = torch.cat([pos_dist, neg_dist], dim=0)
    label = torch.cat(
        [torch.ones(anchor.size()),
         torch.zeros(anchor.size() * 10)])

    loss = cls(dis, label)
    return loss
