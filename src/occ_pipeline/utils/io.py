"""Pickle-free artifacts with atomic replacement and validation."""
import json
import os
from pathlib import Path
import tempfile
import numpy as np


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as f:
        json.dump(value,f,indent=2,allow_nan=False);name=f.name
    os.replace(name,path)


def save_occupancy_npz(path,occupancy,point_counts,metadata,**extras):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('wb',dir=path.parent,delete=False) as f:
        np.savez_compressed(f,occupancy=np.asarray(occupancy,dtype=np.uint8),
                            point_counts=np.asarray(point_counts,dtype=np.uint16),
                            metadata=np.asarray(json.dumps(metadata,sort_keys=True,allow_nan=False)),**extras)
        name=f.name
    os.replace(name,path)


def load_occupancy_npz(path):
    with np.load(path,allow_pickle=False) as d:
        result={'occupancy':d['occupancy'],'point_counts':d['point_counts'],'metadata':json.loads(str(d['metadata'].item()))}
        for key in ('proposal_kind','proposed_occupancy','provenance'):
            if key in d:result[key]=d[key]
    occ=result['occupancy']
    if occ.ndim!=3 or occ.dtype!=np.uint8 or not np.isin(occ,[0,1,2]).all() or result['point_counts'].shape!=occ.shape:
        raise ValueError('Invalid occupancy artifact contract')
    if 'proposal_kind' in result:
        kinds,proposed=result['proposal_kind'],result['proposed_occupancy']
        if kinds.shape!=occ.shape or not np.isin(kinds,[0,4,5]).all() or np.any((kinds!=0)&(occ!=0)):
            raise ValueError('Proposal overwrites measured evidence')
        expected=occ.copy();expected[kinds!=0]=1
        if not np.array_equal(proposed,expected):raise ValueError('Invalid completed occupancy proposal')
    return result
