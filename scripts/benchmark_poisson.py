"""Choose Poisson support on development scans; apply a strict paired-IoU gate.

Poisson proposals are not model inputs or supervision in this pipeline. This
experiment measures geometric completion, separately from learned prediction.
"""
import argparse
from dataclasses import replace
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from occ_pipeline.config import load_config
from occ_pipeline.data.nuscenes_adapter import NuScenesSampleProvider
from occ_pipeline.gt_generation.compare import compare_ground_truth
from occ_pipeline.gt_generation.ray_casting import evidence_grid
from occ_pipeline.gt_generation.voxel_grid import VoxelGridSpec
from occ_pipeline.utils.io import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True);p.add_argument('--plan',required=True)
    p.add_argument('--output',default='artifacts/poisson-benchmark')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    plan=json.loads(Path(a.plan).read_text());c=load_config('configs/poisson_3class.yaml')
    provider=NuScenesSampleProvider(a.data_root,num_sweeps=10,gt_config=c['gt'])
    settings=replace(provider.config,grid=evidence_grid(VoxelGridSpec.from_config(c['grid'])))
    source=provider.source
    rows={split:[source.get('sample',r['sample_token']) for r in plan['selection'][split]] for split in ('dev','test')}
    candidates=[]
    for support in (.25,.5,1.):
        for trim in (.1,.3):
            key=f'support-{support}-trim-{trim}';path=out/f'dev-{key}.json'
            source.samples=lambda split,limit=None: rows['dev'][:limit]
            if path.exists():report=json.loads(path.read_text())
            else:report=compare_ground_truth(source,replace(settings,poisson_support_m=support,poisson_trim_quantile=trim),path,'train',len(rows['dev']),True)
            paired=report['paired_aggregate'];base=paired['box_proposals'];proposed=paired['poisson_proposals']
            n=proposed['samples_scored'];delta=(proposed['occupied_iou']-base['occupied_iou']) if n else None
            per_sample=[]
            for row in report['samples']:
                v=row['variants']
                if all(x['status']=='completed' for x in v.values()):
                    per_sample.append(v['poisson_proposals']['metrics']['occupied_iou']-v['box_proposals']['metrics']['occupied_iou'])
            candidates.append({'support_m':support,'trim_quantile':trim,'paired_samples':n,'iou_delta':delta,
                               'per_sample_deltas':per_sample})
            print(json.dumps({'stage':'development_candidate',**candidates[-1]}),flush=True)
    available=[r for r in candidates if r['iou_delta'] is not None]
    best=max(available,key=lambda r:r['iou_delta']) if available else None
    # Predeclared gate: >=3 paired dev scans, positive pooled delta, no individual harm.
    eligible=bool(best and best['paired_samples']>=3 and best['iou_delta']>0 and all(x>=0 for x in best['per_sample_deltas']))
    decision={'task':'geometric proposal evaluation; not learned-model prediction','candidates':candidates,
              'selected_by_development_only':best,'enable_by_default':eligible,
              'gate':'At least 3 paired development scans, positive pooled occupied-IoU delta and no negative per-scan delta',
              'note':'If the gate fails, retain measured evidence and leave Poisson optional.'}
    if best:
        source.samples=lambda split,limit=None: rows['test'][:limit]
        report=compare_ground_truth(source,replace(settings,poisson_support_m=best['support_m'],poisson_trim_quantile=best['trim_quantile']),
                                    out/'heldout-selected.json','val',len(rows['test']),True)
        decision['heldout_paired_aggregate']=report['paired_aggregate']
        decision['heldout_paired_sample_tokens']=report['paired_sample_tokens']
    write_json(out/'decision.json',decision)
    print(json.dumps({'stage':'complete','enable_by_default':eligible,'selected':best},indent=2),flush=True)


if __name__=='__main__':main()
