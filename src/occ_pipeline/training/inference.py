"""Inference constraints derived solely from the supplied sensor-evidence input."""
import torch
import math


def preserve_input_occupancy(logits,inputs):
    """Never erase an occupied voxel already present in the partial input.

    Only class 1 is constrained. Free/unknown predictions and hidden cells are
    model outputs. This does not read targets or completion proposals.
    """
    if inputs.ndim!=5 or inputs.shape[1]!=1 or logits.ndim!=5 or logits.shape[0]!=inputs.shape[0] or logits.shape[-3:]!=inputs.shape[-3:]:
        raise ValueError('Expected matching [B,C,Z,Y,X] logits and one-channel evidence input')
    occupied=inputs[:,0]>=.5
    if logits.shape[1]==1:
        return torch.where(occupied.unsqueeze(1),torch.maximum(logits,logits.new_tensor(10.)),logits)
    if logits.shape[1]!=3:raise ValueError('Expected one or three output channels')
    result=logits.clone()
    result[:,1]=torch.where(occupied,logits.max(dim=1).values+10.,logits[:,1])
    return result


def apply_inference_decision(logits,inputs,decision=None):
    """Apply settings fitted on development data, without reading targets."""
    if not decision:return logits
    bias=float(decision.get('occupied_bias',0.))
    if not math.isfinite(bias):raise ValueError('Occupied decision bias must be finite')
    result=logits.clone()
    result[:,1 if result.shape[1]==3 else 0]+=bias
    if decision.get('preserve_input',False):result=preserve_input_occupancy(result,inputs)
    return result
