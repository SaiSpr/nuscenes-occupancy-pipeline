"""Small raw-table adapter for the official nuScenes schema.

The public devkit's reverse indices (sample['data']) are not stored on disk.
We reconstruct them from sample_data -> calibrated_sensor -> sensor.
"""

import json
from pathlib import Path

import numpy as np

from occ_pipeline.gt_generation.evidence.geometry import sensor_to_reference, transform_points, rigid_transform, inverse_rigid

CAMERAS = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT")
# Official mini scene split: nuscenes-devkit/python-sdk/nuscenes/utils/splits.py.
MINI_SPLITS = {
    "train": ["scene-0061", "scene-0553", "scene-0655", "scene-0757", "scene-0796", "scene-1077", "scene-1094", "scene-1100"],
    "val": ["scene-0103", "scene-0916"],
}


class NuScenesSource:
    def __init__(self, root: str | Path, version: str = "v1.0-mini", split_file: str | Path | None = None):
        self.root = Path(root).resolve()
        self.version = version
        self.tables = {}
        for name in ("scene", "sample", "sample_data", "sensor", "calibrated_sensor", "ego_pose"):
            path = self.root / version / f"{name}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}. Extract nuScenes metadata and sensor files into the same data root.")
            records = json.loads(path.read_text())
            self.tables[name] = {record["token"]: record for record in records}
            if len(records) != len(self.tables[name]):
                raise ValueError(f"Duplicate token in {name}")
        self.channels = {}
        for record in self.tables["sample_data"].values():
            if not record["is_key_frame"]:
                continue
            calibration = self.get("calibrated_sensor", record["calibrated_sensor_token"])
            channel = self.get("sensor", calibration["sensor_token"])["channel"]
            channels = self.channels.setdefault(record["sample_token"], {})
            if channel in channels:
                raise ValueError(f"Duplicate keyframe channel: {channel}")
            channels[channel] = record
        if split_file:
            self.splits = json.loads(Path(split_file).read_text())
        elif version == "v1.0-mini":
            self.splits = MINI_SPLITS
        else:
            raise ValueError("Pass --splits with explicit train/val scene names for non-mini datasets")
        if set(self.splits) != {"train", "val"} or not all(isinstance(v, list) for v in self.splits.values()):
            raise ValueError("Split file must contain train and val lists of scene names")
        if set(self.splits["train"]) & set(self.splits["val"]):
            raise ValueError("Training and validation scenes must not overlap")
        marker = self.root / "SYNTHETIC_FIXTURE.json"
        self.source_kind = "synthetic_fixture" if marker.exists() else "nuscenes"

    def get(self, table, token):
        try:
            return self.tables[table][token]
        except KeyError as exc:
            raise ValueError(f"Broken nuScenes reference: {table}/{token}") from exc

    def file_path(self, filename: str) -> Path:
        path = (self.root / filename).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Sensor file is outside data root: {filename}")
        if not path.is_file():
            raise FileNotFoundError(f"Sensor file missing: {path}")
        return path

    def samples(self, split: str, limit: int | None = None) -> list[dict]:
        if split not in self.splits or (limit is not None and limit < 1):
            raise ValueError("Expected train/val split and a positive sample limit")
        by_scene = []
        for scene in sorted(self.tables["scene"].values(), key=lambda s: s["name"]):
            if scene["name"] in self.splits[split]:
                by_scene.append(sorted((s for s in self.tables["sample"].values() if s["scene_token"] == scene["token"]), key=lambda s: s["timestamp"]))
        # Round-robin scenes so small demonstrations cover more than one scene.
        selected = [scene[i] for i in range(max(map(len, by_scene), default=0)) for scene in by_scene if i < len(scene)]
        selected = selected[:limit]
        if not selected:
            raise ValueError(f"No samples found for {split}; check dataset version and split file")
        return selected

    def load_sample(self, token: str) -> dict:
        sample = self.get("sample", token)
        channels = self.channels.get(token, {})
        missing = set(("LIDAR_TOP", *CAMERAS)) - set(channels)
        if missing:
            raise ValueError(f"Sample {token} is missing keyframe sensors: {sorted(missing)}")
        lidar = channels["LIDAR_TOP"]
        ref_pose = self.get("ego_pose", lidar["ego_pose_token"])
        lidar_calib = self.get("calibrated_sensor", lidar["calibrated_sensor_token"])
        reference_from_lidar = sensor_to_reference(lidar_calib, ref_pose, ref_pose)
        raw = np.fromfile(self.file_path(lidar["filename"]), dtype="<f4")
        if raw.size % 5:
            raise ValueError("nuScenes LiDAR file must contain float32 x,y,z,intensity,ring records")
        points = transform_points(raw.reshape(-1, 5)[:, :3], reference_from_lidar)
        camera_records = []
        for name in CAMERAS:
            record = channels[name]
            self.file_path(record["filename"])
            calibration = self.get("calibrated_sensor", record["calibrated_sensor_token"])
            pose = self.get("ego_pose", record["ego_pose_token"])
            camera_records.append({
                "channel": name, "filename": record["filename"], "sample_data_token": record["token"],
                "timestamp": record["timestamp"], "width": record["width"], "height": record["height"],
                "intrinsic": calibration["camera_intrinsic"],
                "reference_from_camera": sensor_to_reference(calibration, pose, ref_pose).tolist(),
            })
        return {
            "sample_token": token, "scene_token": sample["scene_token"],
            "scene_name": self.get("scene", sample["scene_token"])["name"],
            "source_kind": self.source_kind, "version": self.version,
            "reference_timestamp": lidar["timestamp"], "reference_frame": "ego_at_lidar_timestamp",
            "reference_ego_pose": ref_pose, "lidar_filename": lidar["filename"],
            "lidar_sample_data_token": lidar["token"], "cameras": camera_records,
            "points": points, "origin": reference_from_lidar[:3, 3].copy(),
        }

    def _load_annotations(self):
        if "sample_annotation" not in self.tables:
            path = self.root / self.version / "sample_annotation.json"
            if not path.is_file():
                raise FileNotFoundError("Temporal GT requires sample_annotation.json; no silent synthetic fallback")
            records = json.loads(path.read_text())
            self.tables["sample_annotation"] = {r["token"]: r for r in records}
            self.annotations_by_sample = {}
            for record in records:
                self.annotations_by_sample.setdefault(record["sample_token"], []).append(record)

    def _boxes_at(self, lidar, reference_from_global):
        from occ_pipeline.gt_generation.evidence.temporal import Box, interpolate_annotation
        self._load_annotations()
        current_sample = self.get("sample", lidar["sample_token"])
        current = self.annotations_by_sample.get(current_sample["token"], [])
        if lidar["is_key_frame"]:
            return [Box.from_annotation(r, reference_from_global) for r in current]
        previous_token = current_sample.get("prev")
        if not previous_token:
            # No bracketing annotations: caller must not silently treat it as static.
            return None
        previous_sample = self.get("sample", previous_token)
        previous = {r["instance_token"]: r for r in self.annotations_by_sample.get(previous_token, [])}
        result = []
        # New/disappearing tracks are uncertain: skip this whole sweep rather than retain trails.
        if set(previous) != {r["instance_token"] for r in current}:
            return None
        for record in current:
            interpolated = interpolate_annotation(previous[record["instance_token"]], record, lidar["timestamp"],
                                                 previous_sample["timestamp"], current_sample["timestamp"])
            if interpolated is None:
                return None
            result.append(Box.from_annotation(interpolated, reference_from_global))
        return result

    def load_sweeps(self, token, config):
        """Past-only sweeps in keyframe ego coordinates, each with its real origin.

        Unbracketed non-keyframes are skipped in motion-compensated mode.
        Reposed returns provide hits only; their invented rays are never carved.
        """
        sample = self.get("sample", token)
        lidar = self.channels[token]["LIDAR_TOP"]
        pose = self.get("ego_pose", lidar["ego_pose_token"])
        reference_from_global = inverse_rigid(rigid_transform(pose["translation"], pose["rotation"]))
        needs_boxes = config.motion_compensation or config.box_fill or config.poisson_enabled
        key_boxes = self._boxes_at(lidar, reference_from_global) if needs_boxes else []
        destinations = {box.instance: box for box in key_boxes}
        sweeps, skipped, seen = [], [], set()
        current = lidar
        for _ in range(config.num_sweeps):
            if current["token"] in seen:
                raise ValueError("Cycle in LiDAR sweep chain")
            seen.add(current["token"])
            linked = self.get("sample", current["sample_token"])
            if linked["scene_token"] != sample["scene_token"]:
                raise ValueError("LiDAR sweep crosses scene boundary")
            if current["timestamp"] > lidar["timestamp"]:
                raise ValueError("Future sweep in past-only GT")
            calibration = self.get("calibrated_sensor", current["calibrated_sensor_token"])
            sensor = self.get("sensor", calibration["sensor_token"])
            if sensor["channel"] != "LIDAR_TOP":
                raise ValueError("Sweep chain changed sensor")
            sweep_pose = self.get("ego_pose", current["ego_pose_token"])
            transform = sensor_to_reference(calibration, sweep_pose, pose)
            raw = np.fromfile(self.file_path(current["filename"]), dtype="<f4")
            if raw.size % 5:
                raise ValueError("nuScenes LiDAR file must contain float32 x,y,z,intensity,ring records")
            xyz = raw.reshape(-1,5)[:, :3]
            finite = np.isfinite(xyz).all(axis=1)
            xyz = xyz[finite]
            ego_removed = 0
            if config.ego_box_min is not None:
                local = transform_points(xyz, rigid_transform(calibration["translation"], calibration["rotation"]))
                body = ((local >= config.ego_box_min) & (local <= config.ego_box_max)).all(axis=1)
                ego_removed = int(body.sum())
                xyz = xyz[~body]
            points = transform_points(xyz, transform)
            origin = transform[:3,3].copy()
            distance = np.linalg.norm(points-origin, axis=1)
            range_keep = (distance >= config.min_range_m) & (distance > 1e-12)
            points, distance = points[range_keep], distance[range_keep]
            boxes = self._boxes_at(current, reference_from_global) if needs_boxes else []
            if boxes is None:
                skipped.append(current["token"])
            else:
                moved = np.zeros(len(points), dtype=bool)
                in_box = np.zeros(len(points), dtype=bool)
                keep = np.ones(len(points), dtype=bool)
                corrected = points.copy()
                for box in sorted(boxes, key=lambda b: b.instance):
                    inside = box.contains(points, config.box_margin_m) & ~in_box
                    in_box |= inside
                    if config.motion_compensation:
                        target = destinations.get(box.instance)
                        if target is None:
                            keep[inside] = False
                        elif not (np.allclose(box.center, target.center, atol=1e-5, rtol=0) and
                                  np.allclose(box.rotation, target.rotation, atol=1e-5, rtol=0)):
                            corrected[inside] = box.reposed(points[inside], target)
                            moved[inside] = True
                sweeps.append({"points": corrected[keep], "origin": origin, "moved": moved[keep],
                               "ranges": distance[keep],
                               "static": ~in_box[keep], "filename": current["filename"],
                               "sample_data_token": current["token"], "timestamp": current["timestamp"],
                               "ego_removed": ego_removed, "reference_from_sensor": transform.tolist()})
            prev = current.get("prev")
            if not prev:
                break
            previous = self.get("sample_data", prev)
            if previous["timestamp"] >= current["timestamp"]:
                raise ValueError("Sweep timestamps must strictly decrease")
            current = previous
        return sweeps, key_boxes, skipped

