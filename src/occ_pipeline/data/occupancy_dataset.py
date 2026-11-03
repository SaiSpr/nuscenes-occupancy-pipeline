"""Deterministic completion inputs; unknown remains distinct from measured free."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from occ_pipeline.utils.io import load_occupancy_npz


class OccupancyDataset(Dataset):
    def __init__(self, gt_dir, num_classes=1, seed=42, training=False, augment=False):
        if num_classes not in (1,3):
            raise ValueError('num_classes must be 1 or 3')
        self.gt_dir=Path(gt_dir)
        self.num_classes,self.seed,self.training,self.augment=num_classes,seed,training,augment
        self.epoch=0
        manifest=self.gt_dir/'manifest.json'
        if manifest.exists():
            entries=json.loads(manifest.read_text())['files']
            self.files=[self.gt_dir/e for e in entries]
            if any(not p.resolve().is_relative_to(self.gt_dir.resolve()) for p in self.files):
                raise ValueError('Manifest path escapes GT directory')
        else:
            self.files=sorted(self.gt_dir.glob('*.npz'))
        if not self.files:
            raise FileNotFoundError(f'No .npz occupancy files found in {self.gt_dir}')
        self.groups=[]
        for path in self.files:
            meta=load_occupancy_npz(path)['metadata']
            group=meta.get('scene_token') or meta.get('scene_name')
            if group is None and meta.get('source')=='synthetic':
                group='synthetic-'+meta.get('sample_token',path.stem)
            self.groups.append(group)

    def __len__(self):
        return len(self.files)

    def __getitem__(self,index):
        artifact=load_occupancy_npz(self.files[index])
        raw=artifact['occupancy'].copy()
        if raw.ndim!=3 or not np.isin(raw,[0,1,2]).all():
            raise ValueError('Expected [Z,Y,X] occupancy with labels 0/1/2')
        # Path-independent stable sample mask. Train changes only at explicit epochs.
        identity=f'{self.seed}:{self.files[index].name}:{self.epoch if self.training else 0}'
        rng=np.random.default_rng(int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8],'little'))
        if self.training and self.augment:
            for axis in (1,2):
                if rng.random()<.5:raw=np.flip(raw,axis).copy()
        occupied=(raw==1).astype('float32')
        model_input=torch.from_numpy(occupied*(rng.random(raw.shape)>=.3)).float().unsqueeze(0)
        target=torch.from_numpy(occupied).unsqueeze(0) if self.num_classes==1 else torch.from_numpy(raw.astype('int64'))
        return {'input':model_input,'target':target,'observed':torch.from_numpy(raw!=0).unsqueeze(0),
                'occ_raw':torch.from_numpy(raw.copy()),'metadata':json.dumps(artifact['metadata'],sort_keys=True),
                'path':str(self.files[index])}
