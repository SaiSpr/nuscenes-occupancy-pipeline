"""Memory-bounded DDA, retaining the branch's 0/1/2 [Z,Y,X] convention."""
import numpy as np
from occ_pipeline.gt_generation.evidence.config import GridSpec,Config
from occ_pipeline.gt_generation.evidence.temporal_gt import generate_temporal_occupancy


def evidence_grid(spec):
    return GridSpec((spec.x_min,spec.y_min,spec.z_min),(spec.x_max,spec.y_max,spec.z_max),spec.voxel_size)


def cast_free_space(points_ego_xyz,spec,origin_xyz=None):
    points=np.asarray(points_ego_xyz,dtype=float)
    points=points[np.isfinite(points).all(1)]
    origin=np.zeros(3) if origin_xyz is None else np.asarray(origin_xyz,dtype=float)
    sweep=dict(points=points,origin=origin,moved=np.zeros(len(points),bool),static=np.ones(len(points),bool))
    result,_=generate_temporal_occupancy([sweep],[],Config(evidence_grid(spec),min_range_m=0.,max_range_m=1e6))
    labels=np.zeros(result.labels.shape,dtype=np.uint8)
    labels[result.labels==1]=1;labels[result.labels==0]=2
    return labels.transpose(2,1,0)
