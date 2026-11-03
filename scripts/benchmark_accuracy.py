"""Predeclared scene-disjoint, repeated-seed occupancy-completion comparison.

Selection and checkpoint choice use training/development scenes only. The
separate official mini_val scenes are evaluated after all model runs finish.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from torch.utils.data import DataLoader
from occ_pipeline.config import load_config
from occ_pipeline.data.nuscenes_adapter import NuScenesSampleProvider
from occ_pipeline.data.occupancy_dataset import OccupancyDataset
from occ_pipeline.gt_generation.generate import generate_ground_truth
from occ_pipeline.gt_generation.voxel_grid import VoxelGridSpec
from occ_pipeline.training.trainer import train,main_output
from occ_pipeline.training.pipeline import build_model
from occ_pipeline.training.metrics import confusion_matrix,metrics_from_confusion
from occ_pipeline.training.inference import preserve_input_occupancy
from occ_pipeline.utils.io import write_json,load_occupancy_npz


def select_records(source):
    by_scene={}
    for split in ('train','val'):
        for r in source.samples(split):by_scene.setdefault(r['scene_token'],[]).append(r)
    names={r['token']:r['name'] for r in source.tables['scene'].values()}
    train_scenes=sorted({r['scene_token'] for r in source.samples('train')},key=names.get)
    test_scenes=sorted({r['scene_token'] for r in source.samples('val')},key=names.get)
    def choose(scenes,n):
        return [rows[i] for scene in scenes for rows in [sorted(by_scene[scene],key=lambda r:r['timestamp'])]
                for i in np.linspace(0,len(rows)-1,n,dtype=int)]
    return {'train':choose(train_scenes[:-1],6),'dev':choose(train_scenes[-1:],6),'test':choose(test_scenes,8)},names


def assert_selection(directory,records):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    actual=[load_occupancy_npz(directory/name)['metadata']['sample_token'] for name in manifest['files']]
    if actual != [r['token'] for r in records]:
        raise ValueError('Artifacts do not match the predeclared sample selection')


def score(model,dataset,preserve=False,occupied_bias=0.):
    total=torch.zeros((3,3),dtype=torch.long);hidden=torch.zeros_like(total);per_scene={}
    model.eval()
    with torch.no_grad():
        for index,b in enumerate(DataLoader(dataset,batch_size=1,shuffle=False)):
            logits=main_output(model(b['input']))
            logits=logits.clone();logits[:,1]+=occupied_bias
            if preserve:logits=preserve_input_occupancy(logits,b['input'])
            cm=confusion_matrix(logits,b['target']).cpu();total+=cm
            pred=logits.argmax(1);mask=b['input'][:,0]<.5;t=b['target']
            hidden+=torch.bincount((t[mask]*3+pred[mask]).long(),minlength=9).reshape(3,3)
            group=dataset.groups[index]
            per_scene.setdefault(group,torch.zeros_like(total));per_scene[group]+=cm
    result=metrics_from_confusion(total);result['hidden_cells']=metrics_from_confusion(hidden)
    result['per_scene']={k:metrics_from_confusion(v) for k,v in per_scene.items()}
    cm=total.numpy();tp=cm[1,1];fp=cm[:,1].sum()-tp;fn=cm[1].sum()-tp
    result['precision_occupied']=float(tp/(tp+fp)) if tp+fp else None
    result['recall_occupied']=float(tp/(tp+fn)) if tp+fn else None
    return result


def select_decision_bias(model,dataset):
    """Tune only on development scenes, with an identical fixed grid for all models."""
    cached=[];model.eval()
    with torch.no_grad():
        for b in DataLoader(dataset,batch_size=1,shuffle=False):
            cached.append((main_output(model(b['input'])),b['input'],b['target']))
    rows=[]
    for bias in np.arange(-4.,1.01,.25):
        matrix=torch.zeros((3,3),dtype=torch.long)
        for raw,inputs,target in cached:
            logits=raw.clone();logits[:,1]+=float(bias)
            matrix+=confusion_matrix(preserve_input_occupancy(logits,inputs),target)
        rows.append({'occupied_bias':float(bias),'development_metrics':metrics_from_confusion(matrix)})
    # Grid order is fixed; ties prefer a conservative, lower occupied score.
    best=max(rows,key=lambda r:r['development_metrics']['iou_occupied'])
    return best,rows


class CopyInput(torch.nn.Module):
    def forward(self,x):
        result=torch.zeros((len(x),3,*x.shape[-3:]))
        result[:,0]=1-x[:,0];result[:,1]=x[:,0]
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True);p.add_argument('--output',default='artifacts/accuracy-benchmark')
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--seeds',default='42,43,44')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    seeds=[int(x) for x in a.seeds.split(',')]
    if a.epochs<1 or len(set(seeds))!=len(seeds):raise ValueError('Use positive epochs and unique seeds')
    c=load_config('configs/temporal_3class.yaml')
    provider=NuScenesSampleProvider(a.data_root,split='train',num_sweeps=10,gt_config=c['gt'])
    selection,names=select_records(provider.source)
    models={'tiny_cnn':{'name':'tiny_3d_cnn','in_channels':1,'out_channels':3,'hidden_channels':4},
            'unet':{'name':'unet_3d','in_channels':1,'out_channels':3,'base_channels':4,'multi_level':False}}
    plan={'task':'GT occupancy mask completion; not raw-sensor prediction','epochs':a.epochs,'seeds':seeds,
          'mask_seed':42,'models':models,'poisson_supervision':False,
          'selection_rule':'Evenly spaced timestamps: 6 frames per training/development scene, 8 per test scene',
          'selection':{split:[{'sample_token':r['token'],'scene_token':r['scene_token'],'scene_name':names[r['scene_token']]} for r in rows]
                       for split,rows in selection.items()},
          'checkpoint_selection':'Minimum development CE loss. Test scenes excluded from optimization and checkpoint selection.',
          'inference_ablations':['raw','preserve known occupied input'],
          'decision_bias_grid':np.arange(-4.,1.01,.25).tolist(),
          'decision_selection':'Maximum development occupied IoU with input preservation; same grid for every model',
          'limitations':['Small nuScenes mini subset','GT-derived mask completion inputs','Prior smoke runs inspected some official validation frames; this is not a pristine external test set']}
    plan_path=out/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:raise ValueError('Existing benchmark has a different plan; choose a new output directory')
    write_json(plan_path,plan)
    # Generate exactly the predeclared samples. Source raw data never enters the repository.
    for split,records in selection.items():
        if not (out/split/'manifest.json').exists():
            provider.max_samples=len(records)
            provider.source.samples=lambda _split,limit=None,rows=records: rows[:limit]
            paths=generate_ground_truth(provider.iter_samples(),VoxelGridSpec.from_config(c['grid']),out/split,enable_ray_casting=True)
            print(json.dumps({'stage':'prepared','split':split,'samples':len(paths)}),flush=True)
        assert_selection(out/split,records)
    for seed in seeds:
        for name,model in models.items():
            folder=out/f'{name}-seed-{seed}';result_path=folder/'training-summary.json'
            if result_path.exists():continue
            settings={'epochs':a.epochs,'lr':1e-3,'batch_size':2,'num_workers':0,'loss':'ce',
                      'lr_scheduler':'cosine','augment':True,'checkpoint_dir':str(folder),
                      'validation_seed':42}
            start=time.perf_counter()
            result=train(out/'train',model,settings,seed=seed,val_dir=out/'dev')
            result['seconds']=time.perf_counter()-start
            write_json(result_path,result)
            print(json.dumps({'stage':'trained','model':name,'seed':seed,'seconds':result['seconds']}),flush=True)
    # Fit decision settings using development data only, before evaluating test.
    development=OccupancyDataset(out/'dev',3,seed=42)
    decisions={}
    for seed in seeds:
        for name,config in models.items():
            checkpoint=out/f'{name}-seed-{seed}'/'checkpoint_best.pt'
            payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
            _,model=build_model(config);model.load_state_dict(payload['model_state_dict'])
            best,grid=select_decision_bias(model,development)
            decisions[f'{name}/{seed}']=best
            write_json(checkpoint.parent/'decision-selection.json',{'selected':best,'grid':grid,'selected_on':'development scenes only'})
            payload['inference_decision']={'occupied_bias':best['occupied_bias'],'preserve_input':True,
                                           'selected_on':'development scenes only'}
            torch.save(payload,checkpoint.parent/'checkpoint_decision.pt')
    # Test is first scored only after all training and development selection finish.
    dataset=OccupancyDataset(out/'test',3,seed=42)
    report={'plan':plan,'copy_input_baseline':score(CopyInput(),dataset),'runs':[]}
    for seed in seeds:
        for name,config in models.items():
            checkpoint=out/f'{name}-seed-{seed}'/'checkpoint_best.pt'
            payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
            _,model=build_model(config);model.load_state_dict(payload['model_state_dict'])
            report['runs'].append({'model':name,'seed':seed,'selected_epoch':payload['epoch'],
                                   'raw':score(model,dataset),'preserve_input':score(model,dataset,True),
                                   'decision':decisions[f'{name}/{seed}'],
                                   'development_selected':score(model,dataset,True,decisions[f'{name}/{seed}']['occupied_bias'])})
    aggregate={}
    for name in models:
        for mode in ('raw','preserve_input','development_selected'):
            rows=[r[mode] for r in report['runs'] if r['model']==name]
            vals=[r['iou_occupied'] for r in rows]
            aggregate[f'{name}/{mode}']={'occupied_iou_mean':float(np.mean(vals)),
                                         'occupied_iou_std':float(np.std(vals,ddof=1)) if len(vals)>1 else None,
                                         'hidden_occupied_iou_mean':float(np.mean([r['hidden_cells']['iou_occupied'] for r in rows])),
                                         'training_seeds':len(vals)}
    report['aggregate']=aggregate
    deltas=[next(r for r in report['runs'] if r['seed']==seed and r['model']=='unet')['preserve_input']['iou_occupied']-
            next(r for r in report['runs'] if r['seed']==seed and r['model']=='tiny_cnn')['raw']['iou_occupied'] for seed in seeds]
    report['paired_iou_deltas']=deltas
    report['paired_selected_iou_deltas']=[
        next(r for r in report['runs'] if r['seed']==seed and r['model']=='unet')['development_selected']['iou_occupied']-
        next(r for r in report['runs'] if r['seed']==seed and r['model']=='tiny_cnn')['development_selected']['iou_occupied']
        for seed in seeds]
    report['improvement_observed_in_every_seed']=all(x>0 for x in deltas)
    write_json(out/'results.json',report)
    print(json.dumps({'stage':'complete','aggregate':aggregate,'paired_iou_deltas':deltas},indent=2),flush=True)


if __name__=='__main__':main()
