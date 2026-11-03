"""Four-frame integration exercise, deliberately selected; not a benchmark."""
import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from occ_pipeline.config import load_config
from occ_pipeline.data.nuscenes_adapter import NuScenesSampleProvider
from occ_pipeline.gt_generation.generate import generate_ground_truth
from occ_pipeline.gt_generation.voxel_grid import VoxelGridSpec
from occ_pipeline.gt_generation.ray_casting import evidence_grid
from occ_pipeline.gt_generation.compare import compare_ground_truth
from occ_pipeline.training.trainer import train
from occ_pipeline.utils.io import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True);p.add_argument('--selection',required=True)
    p.add_argument('--output',required=True);p.add_argument('--include-poisson',action='store_true')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    selection=json.loads(Path(a.selection).read_text())
    c=load_config('configs/poisson_3class.yaml' if a.include_poisson else 'configs/temporal_3class.yaml')
    spec=VoxelGridSpec.from_config(c['grid'])
    providers={}
    for split in ('train','val'):
        provider=NuScenesSampleProvider(a.data_root,split=split,num_sweeps=10,gt_config=c['gt'])
        provider.source.samples=lambda name,limit=None,s=provider.source:[s.get('sample',t) for t in selection['samples'][name]][:limit]
        providers[split]=provider
        generate_ground_truth(provider.iter_samples(),spec,out/split,enable_ray_casting=True)
    write_json(out/'selection.json',selection)
    model=dict(c['model'],base_channels=2)
    settings=dict(c['training'],epochs=1,batch_size=1,checkpoint_dir=str(out/'checkpoints'))
    result=train(out/'train',model,settings,seed=42,val_dir=out/'val')
    write_json(out/'training-summary.json',result)
    module_spec=importlib.util.spec_from_file_location('evaluation',Path(__file__).with_name('evaluate.py'))
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    module.evaluate(out/'checkpoints/checkpoint_best.pt',out/'val',out/'evaluation.json')
    provider=providers['val']
    comparison=compare_ground_truth(provider.source,replace(provider.config,grid=evidence_grid(spec)),
                                    out/'heldout-comparison.json',limit=2,include_poisson=a.include_poisson)
    print(comparison['paired_aggregate'])


if __name__=='__main__':main()
