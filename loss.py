import torch


def triplet_loss(anchor, positive, negative, alpha=0.4, device='cuda'):
    pos_dist = torch.sum((anchor - positive).pow(2), axis=1).mean()
    anchor = anchor.repeat(10, 1)
    neg_dist = torch.sum((anchor - negative).pow(2), axis=1).mean()
    # print(pos_dist, neg_dist)
    basic_loss = pos_dist - neg_dist + alpha
    loss = torch.max(basic_loss, torch.tensor([0], device=device).float())
    return loss
