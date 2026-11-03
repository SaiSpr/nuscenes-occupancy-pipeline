"""Whole-dataset checkpoint evaluation on disjoint scenes, with fixed masks."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from torch.utils.data import DataLoader
from occ_pipeline.data.occupancy_dataset import OccupancyDataset
from occ_pipeline.training.pipeline import build_model,resolve_device
from occ_pipeline.training.trainer import evaluate_loader
from occ_pipeline.utils.io import write_json


def evaluate(checkpoint,gt_dir,output,device='cpu',batch_size=1):
    if batch_size<1:raise ValueError('batch_size must be positive')
    device=resolve_device(device)
    payload=torch.load(checkpoint,map_location=device,weights_only=True)
    protocol=payload.get('protocol')
    if not protocol:raise ValueError('Checkpoint lacks scene/weight protocol; retrain with the updated trainer')
    config=payload['config'];classes=config['model'].get('out_channels',1)
    dataset=OccupancyDataset(gt_dir,classes,seed=protocol.get('validation_seed',protocol['seed']))
    if any(g is None for g in dataset.groups):raise ValueError('Scene metadata is required')
    if set(dataset.groups)&set(protocol['train_scenes']):raise ValueError('Scene leakage: evaluation contains training scenes')
    _,model=build_model(config['model']);model=model.to(device)
    model.load_state_dict(payload['model_state_dict'],strict=True)
    loss,metrics=evaluate_loader(model,DataLoader(dataset,batch_size=batch_size,shuffle=False),device,classes==3,
                                 protocol['fixed_class_weights'],protocol['binary_positive_weight'],
                                 config['training'].get('loss','bce'),payload.get('inference_decision'))
    report={'task':'GT mask completion, not raw sensor prediction','samples_evaluated':len(dataset),
            'scene_disjoint':True,'loss':loss,'metrics':metrics,'seed':protocol['seed'],
            'files':[p.name for p in dataset.files],
            'mask_seed':protocol.get('validation_seed',protocol['seed']),
            'inference_decision':payload.get('inference_decision')}
    write_json(output,report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--gt-dir',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=1)
    a=p.parse_args();print(evaluate(a.checkpoint,a.gt_dir,a.output,a.device,a.batch_size))


if __name__=='__main__':main()
