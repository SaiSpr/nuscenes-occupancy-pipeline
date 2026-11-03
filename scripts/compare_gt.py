import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from occ_pipeline.config import load_config
from occ_pipeline.data.nuscenes_adapter import NuScenesSampleProvider
from occ_pipeline.gt_generation.compare import compare_ground_truth
from occ_pipeline.gt_generation.ray_casting import evidence_grid
from occ_pipeline.gt_generation.voxel_grid import VoxelGridSpec
from dataclasses import replace


def main():
    p=argparse.ArgumentParser(description='Past-only GT vs withheld current scan; not model evaluation')
    p.add_argument('--config',default='configs/temporal_3class.yaml');p.add_argument('--nuscenes-root',required=True)
    p.add_argument('--split',choices=['train','val'],default='val');p.add_argument('--max-samples',type=int,default=24)
    p.add_argument('--include-poisson',action='store_true');p.add_argument('--output',required=True)
    a=p.parse_args();c=load_config(a.config)
    provider=NuScenesSampleProvider(a.nuscenes_root,split=a.split,num_sweeps=c['nuscenes']['num_sweeps'],gt_config=c['gt'])
    settings=replace(provider.config,grid=evidence_grid(VoxelGridSpec.from_config(c['grid'])))
    print(compare_ground_truth(provider.source,settings,a.output,a.split,a.max_samples,a.include_poisson)['paired_aggregate'])


if __name__=='__main__':main()
