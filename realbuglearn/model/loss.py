import torch
import torch.nn as nn
import torch.nn.functional as F

class PtrFocalLoss(nn.Module):

    def __init__(self, gamma=2, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, masks):
        probs = F.softmax(logits, dim=1)
        probs = torch.sum(torch.mul(probs, masks), dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        log_probs = torch.sum(torch.mul(log_probs, masks), dim=1)
        loss = -1 * (1-probs) ** self.gamma * log_probs
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

class FocalLoss(nn.Module):
    # adapted from https://github.com/ShannonAI/dice_loss_for_NLP/blob/master/loss/focal_loss.py

    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, labels):
        labels = labels.unsqueeze(-1)
        probs = F.softmax(logits, dim=1)
        probs = probs.gather(1, labels).squeeze(-1)
        log_probs = F.log_softmax(logits, dim=1)
        log_probs = log_probs.gather(1, labels).squeeze(-1)

        if self.alpha is not None:
            at = self.alpha.gather(0, labels.squeeze(-1))
            log_probs = log_probs * at

        loss = -1 * (1-probs) ** self.gamma * log_probs
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
