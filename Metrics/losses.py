import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = m1 * m2

        score = (
            2.0
            * (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + m2.sum(1) + self.smooth)
        )
        score = 1 - score.sum() / num
        return score


import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        epsilon = 1e-7
        preds = torch.clamp(preds, epsilon, 1.0 - epsilon)  # 防止取对数时出现数值不稳定
        pt = preds * labels + (1 - preds) * (1 - labels)
        alpha_factor = self.alpha * labels + (1 - self.alpha) * (1 - labels)
        focal_weight = alpha_factor * (1.0 - pt).pow(self.gamma)

        loss = -focal_weight * pt.log()
        return torch.mean(loss)