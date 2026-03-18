import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # logits: (N, C), targets: (N, C) in {0,1}
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p*targets + (1-p)*(1-targets)
        alpha_t = self.alpha*targets + (1-self.alpha)*(1-targets)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * ce
        return loss.sum() / (targets.sum().clamp(min=1.0))

class SmoothL1(nn.Module):
    def __init__(self, beta=1.0/9.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        # pred/target: (N,4)
        diff = torch.abs(pred - target)
        loss = torch.where(diff < self.beta, 0.5 * diff**2 / self.beta, diff - 0.5*self.beta)
        return loss.mean()
