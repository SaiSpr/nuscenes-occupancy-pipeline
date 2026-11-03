import torch


def confusion_matrix(logits,target):
    pred=logits.argmax(1)
    return torch.bincount((target.reshape(-1)*3+pred.reshape(-1)).long(),minlength=9).reshape(3,3)


def metrics_from_confusion(matrix):
    matrix=torch.as_tensor(matrix,dtype=torch.float64)
    tp=matrix.diag();den=matrix.sum(0)+matrix.sum(1)-tp
    # Exclude absent classes from mIoU rather than assigning them a perfect score.
    ious=torch.where(den>0,tp/den,torch.nan)
    result={name:float(ious[i]) if den[i]>0 else None for i,name in enumerate(('iou_unknown','iou_occupied','iou_free'))}
    result['mean_iou']=float(ious.nanmean()) if (den>0).any() else None
    result['accuracy']=float(tp.sum()/matrix.sum()) if matrix.sum()>0 else None
    result['confusion_matrix']=matrix.long().tolist()
    return result


def multiclass_occupancy_metrics(logits,target):
    return metrics_from_confusion(confusion_matrix(logits,target))


def binary_occupancy_metrics(logits,target,threshold=.5,observed=None):
    if observed is None:observed=torch.ones_like(target,dtype=torch.bool)
    pred=torch.sigmoid(logits[observed])>=threshold
    truth=target[observed]>=.5
    tp=float((pred&truth).sum());fp=float((pred&~truth).sum());fn=float((~pred&truth).sum());tn=float((~pred&~truth).sum())
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'iou':tp/(tp+fp+fn) if tp+fp+fn else 1.,
            'precision':tp/(tp+fp) if tp+fp else 1.,'recall':tp/(tp+fn) if tp+fn else 1.,
            'target_occupancy_ratio':float(truth.float().mean()) if truth.numel() else 0.,
            'pred_occupancy_ratio':float(pred.float().mean()) if pred.numel() else 0.}
