from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from occ_pipeline.geometry.transforms import invert_rigid_transform, transform_points


@dataclass(frozen=True)
class PointCloudSample:
    sample_token: str
    points_ego_xyz: np.ndarray
    sensor_origin_ego_xyz: np.ndarray  # LiDAR origin in keyframe ego frame
    metadata: dict
    sweeps: tuple = ()
    boxes: tuple = ()
    evidence_config: object = None


class SyntheticSampleProvider:
    """Deterministic local fallback for demos, tests, and walkthrough video setup."""

    def __init__(self, max_samples: int = 5, seed: int = 42) -> None:
        self.max_samples = max_samples
        self.seed = seed

    def iter_samples(self) -> Iterator[PointCloudSample]:
        rng = np.random.default_rng(self.seed)
        for idx in range(self.max_samples):
            # Road-like horizontal returns.
            road_xy = rng.uniform([-30.0, -30.0], [30.0, 30.0], size=(900, 2))
            road_z = rng.normal(loc=-1.45, scale=0.03, size=(900, 1))
            road = np.concatenate([road_xy, road_z], axis=1)

            # A box-like obstacle in front of the vehicle.
            center = np.array([8.0 + idx, 2.0, -0.5], dtype=np.float32)
            obstacle = rng.uniform([-1.5, -0.8, -0.8], [1.5, 0.8, 1.0], size=(350, 3)) + center

            # A vertical wall or barrier.
            wall_y = np.full((250, 1), -12.0 + idx)
            wall_x = rng.uniform(-20.0, 20.0, size=(250, 1))
            wall_z = rng.uniform(-1.5, 1.5, size=(250, 1))
            wall = np.concatenate([wall_x, wall_y, wall_z], axis=1)

            points = np.concatenate([road, obstacle, wall], axis=0).astype(np.float32)
            yield PointCloudSample(
                sample_token=f"synthetic_{idx:04d}",
                points_ego_xyz=points,
                sensor_origin_ego_xyz=np.zeros(3, dtype=np.float32),
                metadata={"source": "synthetic", "sample_index": idx},
            )


class NuScenesSampleProvider:
    """Raw-table reader with per-sweep origins and optional tracked-box correction.

    Missing files fail clearly; unbracketed annotation intervals are reported and
    skipped by the conservative evidence core. No remote devkit is required.
    """
    def __init__(self,dataroot,version="v1.0-mini",split="mini_train",max_samples=5,
                 num_sweeps=1,gt_config=None,split_file=None):
        from occ_pipeline.gt_generation.evidence.nuscenes import NuScenesSource
        from occ_pipeline.gt_generation.evidence.config import Config,GridSpec
        self.source=NuScenesSource(dataroot,version,split_file)
        self.split={"mini_train":"train","mini_val":"val","train":"train","val":"val"}.get(split)
        if self.split is None:raise ValueError("Expected train/val or mini_train/mini_val split")
        self.max_samples=max_samples
        options={k:v for k,v in (gt_config or {}).items() if k not in ("ray_casting","label_policy","include_point_counts")}
        self.config=Config(GridSpec(),num_sweeps=num_sweeps,**options)

    def iter_samples(self):
        import hashlib
        for record in self.source.samples(self.split,self.max_samples):
            sweeps,boxes,skipped=self.source.load_sweeps(record["token"],self.config)
            if not sweeps:raise ValueError("No usable LiDAR sweeps")
            metadata={"source":self.source.source_kind,"version":self.source.version,"split":self.split,
                      "scene_token":record["scene_token"],"sweeps_used":len(sweeps),
                      "skipped_sweeps":skipped,"reference_frame":"ego_at_lidar_timestamp",
                      "sweep_inputs":[{"token":s["sample_data_token"],"timestamp":s["timestamp"],
                                       "origin":s["origin"].tolist(),"transform":s["reference_from_sensor"],
                                       "sha256":hashlib.sha256(self.source.file_path(s["filename"]).read_bytes()).hexdigest()}
                                      for s in sweeps]}
            if boxes or self.config.motion_compensation or self.config.poisson_enabled:
                metadata["annotation_sha256"]=hashlib.sha256((self.source.root/self.source.version/"sample_annotation.json").read_bytes()).hexdigest()
            yield PointCloudSample(record["token"],np.concatenate([s["points"] for s in sweeps]),
                                   sweeps[0]["origin"],metadata,tuple(sweeps),tuple(boxes),self.config)
