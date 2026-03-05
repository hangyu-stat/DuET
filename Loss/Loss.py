import torch.nn as nn
import torch


class bce_loss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super(bce_loss, self).__init__()
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction

    def forward(self, p: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-7
        p = p.clamp(min=eps, max=1 - eps)
        loss = - (target * torch.log(p) +
                  (1.0 - target) * torch.log(1.0 - p))

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def isd_function(x):
    return torch.log(x) + (1.0 - x) / x


class var_loss(nn.Module):
    def __init__(self, reduction: str = "mean", lmd: float = 0.1, img_dec: bool = True):
        super(var_loss, self).__init__()
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction
        self.lmd = lmd
        self.img_dec = img_dec

    def forward(self, alpha_s: torch.Tensor, beta_s: torch.Tensor, alpha_a: torch.Tensor, beta_a: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.img_dec:
            loss = torch.digamma(alpha_s + beta_s) + torch.digamma(alpha_a + beta_a) + target * (self.lmd * isd_function(beta_s) + self.lmd * isd_function(beta_a) - torch.digamma(alpha_s)- torch.digamma(alpha_a))+ (1 - target) * (self.lmd * isd_function(alpha_s) + self.lmd * isd_function(alpha_a) - torch.digamma(beta_s)- torch.digamma(beta_a))
        else:
            loss = torch.digamma(alpha_s + beta_s) + target * (self.lmd * isd_function(beta_s) - torch.digamma(alpha_s))+ (1 - target) * (self.lmd * isd_function(alpha_s) - torch.digamma(beta_s))
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss