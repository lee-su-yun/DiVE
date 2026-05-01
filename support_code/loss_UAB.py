import torch.nn.functional as F


def soft_bce_loss(logit, gt):
    """Soft binary cross entropy between raw logit and a soft target in [0,1]."""
    if logit.dim() == gt.dim() + 1:
        logit = logit.squeeze(1)
    return F.binary_cross_entropy_with_logits(logit, gt, reduction='mean')
