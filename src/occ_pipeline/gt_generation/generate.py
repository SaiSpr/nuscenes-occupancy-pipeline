"""Atomic target generation with manifests and separately stored proposals."""
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import re
import numpy as np
from tqdm import tqdm
from occ_pipeline.gt_generation.ray_casting import cast_free_space,evidence_grid
from occ_pipeline.gt_generation.voxel_grid import voxelize_points
from occ_pipeline.gt_generation.evidence.temporal_gt import generate_temporal_occupancy
from occ_pipeline.utils.io import save_occupancy_npz


def generate_ground_truth(samples,spec,output_dir,min_points_per_voxel=1,enable_ray_casting=False):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    written=[]
    for sample in tqdm(samples,desc='Generating occupancy GT'):
        if not re.fullmatch(r'[A-Za-z0-9_-]+',sample.sample_token):raise ValueError('Unsafe sample token')
        _,counts,metadata=voxelize_points(sample.points_ego_xyz,spec,min_points_per_voxel)
        extras={}
        if sample.sweeps and enable_ray_casting:
            config=replace(sample.evidence_config,grid=evidence_grid(spec),min_hits_per_voxel=min_points_per_voxel)
            result,evidence=generate_temporal_occupancy(sample.sweeps,sample.boxes,config)
            labels=np.zeros(result.labels.shape,dtype=np.uint8)
            labels[result.labels==1]=1;labels[result.labels==0]=2
            occupancy=labels.transpose(2,1,0)
            proposed=occupancy.copy()
            kinds=evidence['proposal_kind'].transpose(2,1,0)
            proposed[kinds!=0]=1
            extras=dict(proposal_kind=kinds,proposed_occupancy=proposed,
                        provenance=evidence['provenance'].transpose(2,1,0))
            metadata.update(stats=result.stats,evidence_config=config.to_dict(),
                            proposals_supervised=False,provenance_values={'unknown':0,'ray_free':1,'hit':2,'reposed_hit':3},
                            proposal_values={'none':0,'box_prior':4,'poisson_shell':5})
        elif enable_ray_casting:occupancy=cast_free_space(sample.points_ego_xyz,spec,sample.sensor_origin_ego_xyz)
        else:
            if sample.evidence_config and (sample.evidence_config.box_fill or sample.evidence_config.poisson_enabled):
                raise ValueError('Reconstruction requires ray_casting=true to preserve free/unknown evidence')
            occupancy=(counts>=min_points_per_voxel).astype(np.uint8)
        metadata.update(sample.metadata)
        metadata.update(sample_token=sample.sample_token,ray_casting=enable_ray_casting,
                        label_policy='0=unknown,1=occupied,2=free',task='partial sensor evidence; completion proposals separate')
        digest=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()[:16]
        metadata['generation_fingerprint']=digest
        out=output_dir/f'{sample.sample_token}-{digest}.npz'
        save_occupancy_npz(out,occupancy,counts,metadata,**extras);written.append(out)
    if not written:raise ValueError('No GT samples generated')
    from occ_pipeline.utils.io import write_json
    write_json(output_dir/'manifest.json',{'files':[p.name for p in written], 'axis_order':'ZYX',
                                         'labels':{'unknown':0,'occupied':1,'free':2}})
    return written
