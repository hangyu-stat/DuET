import torch.nn as nn
import torch


class bce_loss(nn.Module):
    """
    二分类交叉熵:
    input:  已经是概率, 取值在 [0, 1]
    target: 0 或 1 (float 型)，shape 与 input 相同
    """
    def __init__(self, reduction: str = "mean"):
        super(bce_loss, self).__init__()
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction

    def forward(self, p: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 避免 log(0)
        eps = 1e-7
        p = p.clamp(min=eps, max=1 - eps)

        # BCE = -[y*log(p) + (1-y)*log(1-p)]
        loss = - (target * torch.log(p) +
                  (1.0 - target) * torch.log(1.0 - p))

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss


def isd_function(x):
    return torch.log(x) + (1.0 - x) / x


class var_loss(nn.Module):
    """
    变分损失:
    input:  a和s的alpha、beta以及相应的加权系数w
    target: 0 或 1 (float 型)，shape 与 input 相同
    """
    def __init__(self, reduction: str = "mean", lmd: float = 0.1, img_dec: bool = True):
        super(var_loss, self).__init__()
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction
        self.lmd = lmd
        self.img_dec = img_dec

    def forward(self, alpha_s: torch.Tensor, beta_s: torch.Tensor, alpha_a: torch.Tensor, beta_a: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.img_dec:
            # W = (alpha_s + beta_s) / (alpha_s + beta_s + alpha_a + beta_a)
            # W = 0.5
            loss = torch.digamma(alpha_s + beta_s) + torch.digamma(alpha_a + beta_a) + target * (self.lmd * isd_function(beta_s) + self.lmd * isd_function(beta_a) - torch.digamma(alpha_s)- torch.digamma(alpha_a))+ (1 - target) * (self.lmd * isd_function(alpha_s) + self.lmd * isd_function(alpha_a) - torch.digamma(beta_s)- torch.digamma(beta_a))
        else:
            loss = torch.digamma(alpha_s + beta_s) + target * (self.lmd * isd_function(beta_s) - torch.digamma(alpha_s))+ (1 - target) * (self.lmd * isd_function(alpha_s) - torch.digamma(beta_s))
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss