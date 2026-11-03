"""Paired motion-only vs motion+Poisson comparison, with no annotation box fill."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from occ_pipeline.config import load_config
from occ_pipeline.data.nuscenes_adapter import NuScenesSampleProvider
from occ_pipeline.gt_generation.ray_casting import evidence_grid
from occ_pipeline.gt_generation.voxel_grid import VoxelGridSpec
from occ_pipeline.gt_generation.evidence.temporal_gt import generate_temporal_occupancy
from occ_pipeline.gt_generation.compare import compare_labels,counts_to_metrics
from occ_pipeline.utils.io import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True);p.add_argument('--plan',required=True)
    p.add_argument('--output',default='artifacts/combined-validation/comparison.json')
    a=p.parse_args();c=load_config('configs/submission.yaml')
    provider=NuScenesSampleProvider(a.data_root,num_sweeps=10,gt_config=c['gt'])
    source=provider.source;settings=replace(provider.config,grid=evidence_grid(VoxelGridSpec.from_config(c['grid'])))
    selection=json.loads(Path(a.plan).read_text())['selection']['test'];rows=[]
    for r in selection:
        token=r['sample_token'];current=source.channels[token]['LIDAR_TOP']['token']
        ref_config=replace(settings,num_sweeps=1,motion_compensation=False,box_fill=False,poisson_enabled=False)
        ref_sweeps,_,_=source.load_sweeps(token,ref_config)
        reference,_=generate_temporal_occupancy(ref_sweeps,[],ref_config)
        sweeps,boxes,skipped=source.load_sweeps(token,settings)
        past=[s for s in sweeps if s['sample_data_token']!=current]
        if not past:
            rows.append({'sample_token':token,'status':'skipped_no_usable_past_sweeps','skipped':skipped});continue
        measured,evidence=generate_temporal_occupancy(past,boxes,settings)
        rows.append({'sample_token':token,'status':'completed','held_out_token':current,
                     'input_tokens':[s['sample_data_token'] for s in past],
                     'motion_only':compare_labels(measured.labels,reference.labels),
                     'motion_plus_poisson':compare_labels(evidence['proposed_labels'],reference.labels),
                     'poisson_stats':measured.stats['poisson']})
    aggregates={}
    for variant in ('motion_only','motion_plus_poisson'):
        values=[row[variant] for row in rows if row['status']=='completed']
        sums={k:sum(row[k] for row in values) for k in ('tp','fp','fn','wrongly_free','reference_observed','prediction_known_on_reference')}
        aggregates[variant]={**counts_to_metrics(sums.pop('tp'),sums.pop('fp'),sums.pop('fn')),**sums,'samples_scored':len(values)}
    report={'profile':'configs/submission.yaml','config':settings.to_dict(),'box_fill':False,
            'protocol':'Current scan withheld; both variants use exactly the same past sweeps and motion correction.',
            'proposals_used_in_model_supervision':False,'samples':rows,'aggregate':aggregates,
            'limitation':'Sparse geometric agreement on a mini subset, not model accuracy or complete-scene truth.'}
    write_json(a.output,report);print(json.dumps(aggregates,indent=2))


if __name__=='__main__':main()
