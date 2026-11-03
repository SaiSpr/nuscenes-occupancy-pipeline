"""Small real-data GT + model/loss demonstration of the combined profile."""
import argparse
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from occ_pipeline.config import load_config
from occ_pipeline.data.nuscenes_adapter import NuScenesSampleProvider
from occ_pipeline.gt_generation.generate import generate_ground_truth
from occ_pipeline.gt_generation.voxel_grid import VoxelGridSpec
from occ_pipeline.training.pipeline import run_forward_loss_demo
from occ_pipeline.utils.seed import set_seed
from occ_pipeline.utils.io import load_occupancy_npz,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--sample-tokens',required=True,help='Comma-separated keyframe tokens for a reproducible demonstration')
    a=p.parse_args();out=Path(a.output);c=load_config('configs/submission.yaml');set_seed(c['project']['seed'])
    provider=NuScenesSampleProvider(a.data_root,split='val',num_sweeps=c['nuscenes']['num_sweeps'],gt_config=c['gt'])
    tokens=a.sample_tokens.split(',')
    provider.source.samples=lambda split,limit=None:[provider.source.get('sample',t) for t in tokens]
    paths=generate_ground_truth(provider.iter_samples(),VoxelGridSpec.from_config(c['grid']),out/'targets',enable_ray_casting=True)
    summaries=[]
    for path in paths:
        d=load_occupancy_npz(path);meta=d['metadata']
        assert meta['evidence_config']['motion_compensation'] and meta['evidence_config']['poisson_enabled']
        assert not ((d['proposal_kind']!=0)&(d['occupancy']!=0)).any()
        summaries.append({'sample_token':meta['sample_token'],'stats':meta['stats']})
    summary={'profile':'configs/submission.yaml','motion_compensation':True,'poisson_enabled':True,'samples':summaries,
             'learning':run_forward_loss_demo(out/'targets',c['model'],batch_size=1)}
    write_json(out/'demo-summary.json',summary)
    # Visible distinction between measured evidence and inferred proposals.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    d=load_occupancy_npz(paths[0]);occ=d['occupancy'];proposed=d['proposed_occupancy']
    bev=lambda x:np.where((x==1).any(0),1,np.where((x==2).any(0),2,0))
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    cmap=ListedColormap(['#1f2b3e','#f1a33c','#bad7e1'])
    for ax,array,title in zip(axes,[occ,proposed],['Measured motion-corrected labels','Measured labels + Poisson proposals']):
        ax.imshow(bev(array),origin='lower',cmap=cmap,vmin=0,vmax=2);ax.set_title(title,fontsize=10)
        ax.set_xlabel('X voxel');ax.set_ylabel('Y voxel')
    fig.suptitle('Real nuScenes | inferred proposals remain separate from training targets')
    fig.savefig(out/'combined-gt.png',dpi=150);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
