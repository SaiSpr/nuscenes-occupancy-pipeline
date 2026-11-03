from pathlib import Path
import importlib.util
import json
import numpy as np
import pytest
import torch
from occ_pipeline.data.nuscenes_adapter import SyntheticSampleProvider
from occ_pipeline.data.occupancy_dataset import OccupancyDataset
from occ_pipeline.gt_generation.generate import generate_ground_truth
from occ_pipeline.gt_generation.voxel_grid import VoxelGridSpec
from occ_pipeline.models.occupancy_unet import OccupancyUNet3D
from occ_pipeline.training.losses import occupancy_bce_loss,occupancy_ce_loss
from occ_pipeline.training.metrics import confusion_matrix,metrics_from_confusion
from occ_pipeline.training.trainer import scene_indices,train,fixed_class_weights
from occ_pipeline.utils.io import load_occupancy_npz,save_occupancy_npz


SPEC=VoxelGridSpec(-10,10,-10,10,-2,2,1.)


def targets(path,n=4):
    generate_ground_truth(SyntheticSampleProvider(n,7).iter_samples(),SPEC,path,enable_ray_casting=True)
    return path


def evaluate_function():
    p=Path(__file__).resolve().parents[1]/'scripts/evaluate.py'
    spec=importlib.util.spec_from_file_location('branch_evaluate',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m.evaluate


def test_validation_mask_stable_and_proposals_not_inputs(tmp_path):
    p=targets(tmp_path/'gt')
    a=OccupancyDataset(p,3,seed=42);b=OccupancyDataset(p,3,seed=42)
    torch.testing.assert_close(a[0]['input'],a[0]['input'])
    torch.testing.assert_close(a[0]['input'],b[0]['input'])
    # Change only the saved proposal; model input and target must not change.
    art=load_occupancy_npz(a.files[0]);raw=art['occupancy'];kinds=np.zeros(raw.shape,np.uint8)
    kinds[raw==0]=5;proposal=raw.copy();proposal[kinds!=0]=1
    before=a[0]
    save_occupancy_npz(a.files[0],raw,art['point_counts'],art['metadata'],proposal_kind=kinds,proposed_occupancy=proposal)
    torch.testing.assert_close(before['input'],a[0]['input']);torch.testing.assert_close(before['target'],a[0]['target'])


def test_unknown_binary_cells_have_zero_gradient():
    logits=torch.zeros((1,1,1,1,3),requires_grad=True)
    target=torch.tensor([[[[[0.,1.,0.]]]]]);mask=torch.tensor([[[[[False,True,True]]]]])
    occupancy_bce_loss(logits,target,mask).backward()
    assert logits.grad.flatten()[0]==0
    assert logits.grad.flatten()[1]!=0


def test_multiclass_confusion_accumulates_instead_of_averaging_batches():
    logits=torch.tensor([[[[[9.,0.,0.]]],[[[0.,9.,0.]]],[[[0.,0.,9.]]]]])
    target=torch.tensor([[[[0,1,1]]]])
    matrix=confusion_matrix(logits,target)
    m=metrics_from_confusion(matrix)
    assert matrix.tolist()==[[1,0,0],[0,1,1],[0,0,0]]
    assert m['iou_occupied']==.5 and m['iou_free']==0
    assert m['mean_iou']==pytest.approx(.5)
    assert m['iou_free']!=1 # ground-truth free labels are never copied


def test_scene_groups_never_split_neighbouring_frames(tmp_path):
    p=targets(tmp_path/'gt')
    ds=OccupancyDataset(p)
    ds.groups=['A','A','B','B']
    tr,va=scene_indices(ds,.5,42)
    assert {ds.groups[i] for i in tr}.isdisjoint({ds.groups[i] for i in va})
    ds.groups=['A']*4
    with pytest.raises(ValueError,match='Only one scene'):scene_indices(ds,.2,42)


def test_manifest_excludes_stale_targets(tmp_path):
    p=targets(tmp_path/'gt',3)
    old=set(p.glob('*.npz'))
    targets(p,1)
    assert len(OccupancyDataset(p))==1
    assert len(set(p.glob('*.npz'))|old)>=3


def test_auxiliary_unet_binary_and_three_class_shapes():
    for channels in (1,3):
        net=OccupancyUNet3D(out_channels=channels,base_channels=2,multi_level=True)
        main,aux=net(torch.randn(1,1,3,17,19))
        assert main.shape==(1,channels,3,17,19)
        assert len(aux)==2


def test_three_class_training_and_full_evaluation_are_repeatable(tmp_path):
    p=targets(tmp_path/'gt',4)
    result=train(p,{'name':'unet_3d','in_channels':1,'out_channels':3,'base_channels':2,'multi_level':True},
                 {'epochs':1,'lr':1e-3,'val_split':.5,'batch_size':2,'checkpoint_dir':str(tmp_path/'ckpt'),'loss':'ce'},seed=42)
    assert result['train_samples']==2 and result['val_samples']==2
    assert result['history']['val_metrics'][0]['confusion_matrix']
    checkpoint=tmp_path/'ckpt/checkpoint_best.pt'
    # Explicit val-only directory from the recorded split.
    val=tmp_path/'val';val.mkdir()
    for name in result['protocol']['val_files']:
        (val/name).write_bytes((p/name).read_bytes())
    evaluate=evaluate_function()
    a=evaluate(checkpoint,val,tmp_path/'a.json',batch_size=1)
    b=evaluate(checkpoint,val,tmp_path/'b.json',batch_size=2)
    assert a['samples_evaluated']==2 and a['metrics']==b['metrics']
    assert a['loss']==pytest.approx(b['loss'],abs=1e-5)
    with pytest.raises(ValueError,match='Scene leakage'):
        evaluate(checkpoint,p,tmp_path/'bad.json')


def test_ce_uses_supplied_fixed_weights():
    logits=torch.randn(2,3,2,2,2,requires_grad=True);target=torch.randint(0,3,(2,2,2,2))
    weights=[.1,2.,.9]
    torch.testing.assert_close(occupancy_ce_loss(logits,target,weights),
                              torch.nn.functional.cross_entropy(logits,target,weight=torch.tensor(weights)))


def test_minimum_hit_counts_do_not_overflow_uint16():
    from occ_pipeline.gt_generation.voxel_grid import voxelize_points
    _,counts,_=voxelize_points(np.zeros((70000,3)),SPEC)
    assert counts.max()==65535


def test_actual_poisson_backend_retains_measured_evidence():
    # An absent optional backend can skip; an installed but broken backend must fail.
    pytest.importorskip('open3d', exc_type=ModuleNotFoundError)
    from dataclasses import replace
    from occ_pipeline.gt_generation.evidence.config import Config,GridSpec
    from occ_pipeline.gt_generation.evidence.temporal_gt import generate_temporal_occupancy
    rng=np.random.default_rng(7);points=rng.normal(size=(1000,3));points=points/np.linalg.norm(points,axis=1)[:,None]*3
    sweep=dict(points=points,origin=np.zeros(3),moved=np.zeros(len(points),bool),static=np.ones(len(points),bool))
    cfg=Config(GridSpec((-5,-5,-5),(5,5,5),.5),min_range_m=0.,poisson_depth=5,poisson_enabled=True)
    a,evidence=generate_temporal_occupancy([sweep],[],cfg)
    b,_=generate_temporal_occupancy([sweep],[],replace(cfg,poisson_enabled=False))
    np.testing.assert_array_equal(a.labels,b.labels)
    assert a.stats['poisson']['surface_proposals']>0
    assert not evidence['proposal_kind'][a.observed].any()


def test_inference_preserves_input_without_looking_at_targets():
    from occ_pipeline.training.inference import preserve_input_occupancy,apply_inference_decision
    inputs=torch.tensor([[[[[1.,0.,0.]]]]])
    logits=torch.tensor([[[[[5.,5.,0.]]],[[[-4.,1.,0.]]],[[[2.,0.,5.]]]]])
    corrected=preserve_input_occupancy(logits,inputs)
    assert corrected.argmax(1).flatten().tolist()==[1,0,2]
    torch.testing.assert_close(corrected[:,:,:, :,1:],logits[:,:,:, :,1:])
    # A negative occupied decision bias changes only the selected output channel.
    adjusted=apply_inference_decision(logits,inputs,{'occupied_bias':-1.,'preserve_input':False})
    torch.testing.assert_close(adjusted[:,0],logits[:,0]);torch.testing.assert_close(adjusted[:,2],logits[:,2])
    with pytest.raises(ValueError,match='finite'):
        apply_inference_decision(logits,inputs,{'occupied_bias':float('nan')})


def test_benchmark_rejects_truncated_sample_selection(tmp_path):
    p=targets(tmp_path/'gt',3)
    spec=importlib.util.spec_from_file_location('benchmark_accuracy',Path(__file__).resolve().parents[1]/'scripts/benchmark_accuracy.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    files=json.loads((p/'manifest.json').read_text())['files']
    records=[{'token':load_occupancy_npz(p/name)['metadata']['sample_token']} for name in files]
    module.assert_selection(p,records)
    with pytest.raises(ValueError,match='predeclared'):
        module.assert_selection(p,records+[{'token':'missing-planned-sample'}])


def test_checkpoint_decision_applied_by_evaluation_loader():
    from occ_pipeline.training.trainer import evaluate_loader
    class Model(torch.nn.Module):
        def forward(self,x):
            logits=torch.zeros((len(x),3,*x.shape[-3:]));logits[:,0]=5
            return logits
    batch={'input':torch.tensor([[[[[1.,0.,0.]]]]]),'target':torch.tensor([[[[1,0,2]]]]),
           'observed':torch.tensor([[[[[True,False,True]]]]])}
    raw=evaluate_loader(Model(),[batch],'cpu',True,[1.,1.,1.],1.)[1]
    fixed=evaluate_loader(Model(),[batch],'cpu',True,[1.,1.,1.],1.,inference_decision={'preserve_input':True})[1]
    assert raw['iou_occupied']==0 and fixed['iou_occupied']==1
    assert fixed['confusion_matrix']==[[1,0,0],[0,1,0],[1,0,0]]


def test_submission_profile_keeps_motion_and_poisson_enabled():
    from occ_pipeline.config import load_config
    config=load_config(Path(__file__).resolve().parents[1]/'configs/submission.yaml')
    assert config['gt']['motion_compensation'] is True
    assert config['gt']['poisson_enabled'] is True
    assert config['gt']['box_fill'] is False
