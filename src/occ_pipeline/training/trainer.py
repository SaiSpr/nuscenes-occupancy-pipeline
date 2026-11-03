"""Scene-disjoint training and whole-dataset evaluation for completion models."""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset
from occ_pipeline.data.occupancy_dataset import OccupancyDataset
from occ_pipeline.training.losses import occupancy_bce_loss,occupancy_ce_loss,occupancy_focal_loss
from occ_pipeline.training.metrics import binary_occupancy_metrics,confusion_matrix,metrics_from_confusion
from occ_pipeline.training.pipeline import build_model,resolve_device
from occ_pipeline.utils.io import load_occupancy_npz
from occ_pipeline.utils.seed import set_seed


def scene_indices(dataset,val_split,seed):
    if any(g is None for g in dataset.groups):
        raise ValueError('Scene metadata is required for splitting real GT; regenerate targets')
    groups=sorted(set(dataset.groups))
    if not 0<=val_split<1:raise ValueError('val_split must be in [0,1)')
    if val_split==0 or len(groups)==1:
        if len(groups)==1 and len(dataset)>1 and val_split>0:
            raise ValueError('Only one scene: supply --val-dir or explicitly set val_split=0')
        return list(range(len(dataset))),[]
    order=np.random.default_rng(seed).permutation(len(groups))
    n=max(1,min(len(groups)-1,int(len(groups)*val_split)))
    val={groups[i] for i in order[:n]}
    return [i for i,g in enumerate(dataset.groups) if g not in val],[i for i,g in enumerate(dataset.groups) if g in val]


def fixed_class_weights(dataset,indices):
    counts=np.zeros(3,dtype=np.float64)
    for i in indices:
        counts+=np.bincount(load_occupancy_npz(dataset.files[i])['occupancy'].reshape(-1),minlength=3)
    inv=1/np.maximum(counts,1)
    return (inv/inv.sum()*3).tolist(),float(np.clip(counts[2]/max(counts[1],1),1,100))


def main_output(output):
    return output[0] if isinstance(output,tuple) else output


def objective(output,batch,multiclass,weights,pos_weight,loss_name='bce'):
    target,observed=batch['target'],batch['observed']
    def criterion(logits,t,mask):
        if multiclass:return occupancy_ce_loss(logits,t,weights)
        if loss_name=='focal':return occupancy_focal_loss(logits,t,mask)
        return occupancy_bce_loss(logits,t,mask,pos_weight)
    logits=main_output(output)
    loss=criterion(logits,target,observed)
    if isinstance(output,tuple):
        for factor,aux in zip((.4,.2),output[1]):
            t=F.interpolate(target.unsqueeze(1).float() if multiclass else target,
                            size=aux.shape[-3:],mode='nearest')
            if multiclass:t=t[:,0].long()
            mask=F.interpolate(observed.float(),size=aux.shape[-3:],mode='nearest').bool()
            loss=loss+factor*criterion(aux,t,mask)
    return loss


def evaluate_loader(model,loader,device,multiclass,weights,pos_weight,loss_name='bce',inference_decision=None):
    model.eval();matrix=torch.zeros((3,3),dtype=torch.long)
    counts=dict(tp=0.,fp=0.,fn=0.,tn=0.)
    numerator,denominator=0.,0.
    with torch.no_grad():
        for raw in loader:
            batch={k:raw[k].to(device) for k in ('input','target','observed')}
            output=model(batch['input']);logits=main_output(output)
            if inference_decision:
                from occ_pipeline.training.inference import apply_inference_decision
                logits=apply_inference_decision(logits,batch['input'],inference_decision)
            # Report main-head loss with a globally consistent denominator.
            loss=objective(logits,batch,multiclass,weights,pos_weight,loss_name)
            if multiclass:
                matrix+=confusion_matrix(logits,batch['target']).cpu()
                n=float(torch.as_tensor(weights,device=device)[batch['target']].sum())
            else:
                m=binary_occupancy_metrics(logits,batch['target'],observed=batch['observed'])
                for k in counts:counts[k]+=m[k]
                n=float(batch['observed'].sum())
            numerator+=float(loss)*n;denominator+=n
    if multiclass:metrics=metrics_from_confusion(matrix)
    else:
        tp,fp,fn=counts['tp'],counts['fp'],counts['fn']
        metrics={**counts,'iou_occupied':tp/(tp+fp+fn) if tp+fp+fn else None,
                 'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None}
    return numerator/denominator if denominator else None,metrics


def train(gt_dir,model_config,training_config,device='cpu',seed=42,val_dir=None):
    set_seed(seed);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    device=resolve_device(device)
    classes=int(model_config.get('out_channels',1));multiclass=classes==3
    dataset=OccupancyDataset(gt_dir,classes,seed,training=True,augment=training_config.get('augment',False))
    validation_seed=int(training_config.get('validation_seed',seed))
    validation=OccupancyDataset(val_dir or gt_dir,classes,validation_seed,training=False)
    if val_dir:
        if any(g is None for g in dataset.groups+validation.groups):raise ValueError('Scene metadata required for external validation')
        if set(dataset.groups)&set(validation.groups):raise ValueError('Scene leakage between training and validation')
        train_ids,val_ids=list(range(len(dataset))),list(range(len(validation)))
    else:train_ids,val_ids=scene_indices(dataset,float(training_config.get('val_split',.2)),seed)
    if not train_ids:raise ValueError('No training samples')
    weights,pos_weight=fixed_class_weights(dataset,train_ids)
    batch_size=int(training_config.get('batch_size',1));workers=int(training_config.get('num_workers',0))
    loader=DataLoader(Subset(dataset,train_ids),batch_size=batch_size,shuffle=True,num_workers=workers,
                      generator=torch.Generator().manual_seed(seed))
    val_loader=DataLoader(Subset(validation,val_ids),batch_size=batch_size,shuffle=False,num_workers=workers) if val_ids else None
    name,model=build_model(dict(model_config));model=model.to(device)
    epochs=int(training_config.get('epochs',10));lr=float(training_config.get('lr',1e-3))
    if epochs<1 or lr<=0:raise ValueError('epochs and lr must be positive')
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    schedule=training_config.get('lr_scheduler','none')
    scheduler=(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=lr*.01) if schedule=='cosine' else
               torch.optim.lr_scheduler.StepLR(optimizer,max(1,epochs//3),gamma=.3) if schedule=='step' else None)
    if schedule not in ('none','cosine','step'):raise ValueError('Unknown lr_scheduler')
    loss_name=training_config.get('loss','bce')
    if loss_name not in ('bce','focal','ce'):raise ValueError('Unknown loss')
    if multiclass and loss_name=='focal':raise ValueError('Three-class mode uses CE; focal currently supports binary only')
    checkpoint_dir=Path(training_config.get('checkpoint_dir','artifacts/checkpoints'));checkpoint_dir.mkdir(parents=True,exist_ok=True)
    history={'train_loss':[],'val_loss':[],'val_iou':[],'val_metrics':[]};best=float('inf')
    protocol={'task':'GT mask completion, not raw-sensor prediction','seed':seed,'validation_seed':validation_seed,'scene_disjoint':True,
              'train_scenes':sorted({dataset.groups[i] for i in train_ids}),
              'val_scenes':sorted({validation.groups[i] for i in val_ids}),
              'train_files':[dataset.files[i].name for i in train_ids],
              'val_files':[validation.files[i].name for i in val_ids],
              'fixed_class_weights':weights,'binary_positive_weight':pos_weight}
    for epoch in range(1,epochs+1):
        dataset.epoch=epoch;model.train();total=0.;batches=0
        for raw in loader:
            batch={k:raw[k].to(device) for k in ('input','target','observed')}
            optimizer.zero_grad();output=model(batch['input'])
            loss=objective(output,batch,multiclass,weights,pos_weight,loss_name)
            if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
            loss.backward();optimizer.step();total+=float(loss.detach());batches+=1
        val_loss,metrics=(evaluate_loader(model,val_loader,device,multiclass,weights,pos_weight,loss_name)
                          if val_loader is not None else (None,{}))
        history['train_loss'].append(total/batches);history['val_loss'].append(val_loss)
        history['val_iou'].append(metrics.get('iou_occupied'));history['val_metrics'].append(metrics)
        payload={'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                 'val_loss':val_loss,'config':{'model':model_config,'training':training_config},'protocol':protocol}
        if val_loss is not None and val_loss<best:
            best=val_loss;torch.save(payload,checkpoint_dir/'checkpoint_best.pt')
        torch.save(payload,checkpoint_dir/'checkpoint_latest.pt')
        if scheduler is not None:scheduler.step()
        print(f'Epoch {epoch}/{epochs} | train_loss={total/batches:.5f} | val_loss={val_loss} | occupied_iou={metrics.get("iou_occupied")}')
    return {'model_name':name,'epochs':epochs,'train_samples':len(train_ids),'val_samples':len(val_ids),
            'history':history,'checkpoint_dir':str(checkpoint_dir),'protocol':protocol}
