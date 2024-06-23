import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
import numpy as np


def loss_fn(logits, label, loss_config):
    if not loss_config or loss_config['name'] == 'ce':
        # cross entropy
        loss = F.binary_cross_entropy_with_logits(logits.contiguous().view(-1), label.float().view(-1))
    else:
        # focal loss
        if loss_config['name'] == "focal":
            ori_loss = F.binary_cross_entropy_with_logits(logits.reshape(-1), label.float().reshape(-1), reduction='none')
            pred_sigmoid = logits.view(-1).sigmoid()
            target = label.float().view(-1)
            pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
            focal_weight = (loss_config['alpha'] * target + (1 - loss_config['alpha']) * (1 - target)) * pt.pow(loss_config['gamma'])
            loss = torch.mean(ori_loss * focal_weight)        
    return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def focalLoss(inputs,targets):
    x = FocalLoss()
    loss = x(inputs,targets)
    return loss