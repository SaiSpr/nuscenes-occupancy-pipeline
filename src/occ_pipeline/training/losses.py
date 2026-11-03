import torch
from torch.nn import functional as F


def occupancy_bce_loss(logits,target,observed=None,pos_weight=None):
    if observed is None:observed=torch.ones_like(target,dtype=torch.bool)
    if not observed.any():return logits.sum()*0
    if pos_weight is None:
        positives=target[observed].sum()
        negatives=observed.sum()-positives
        pos_weight=(negatives/positives.clamp_min(1)).clamp(1,100)
    weight=torch.as_tensor(pos_weight,device=logits.device,dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits[observed],target[observed],pos_weight=weight)


def occupancy_focal_loss(logits,target,observed=None,gamma=2.,alpha=.75):
    if observed is None:observed=torch.ones_like(target,dtype=torch.bool)
    if not observed.any():return logits.sum()*0
    p=torch.sigmoid(logits[observed]);t=target[observed]
    pt=p*t+(1-p)*(1-t)
    return ((alpha*t+(1-alpha)*(1-t))*(1-pt)**gamma*
            F.binary_cross_entropy_with_logits(logits[observed],t,reduction='none')).mean()


def occupancy_ce_loss(logits,target,weights=None):
    """Genuine three-class head. Use fixed training-set weights, never batch frequencies."""
    weight=None if weights is None else torch.as_tensor(weights,device=logits.device,dtype=logits.dtype)
    return F.cross_entropy(logits,target,weight=weight)
