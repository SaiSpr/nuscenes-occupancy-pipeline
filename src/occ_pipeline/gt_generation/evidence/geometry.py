"""Rigid transforms. T_destination_source maps source points into destination."""

import numpy as np


def rigid_transform(translation, quaternion_wxyz) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64)
    if q.shape != (4,) or t.shape != (3,) or not np.isfinite(q).all() or not np.isfinite(t).all():
        raise ValueError("Invalid rigid transform")
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Zero quaternion")
    w, x, y, z = q / norm
    result = np.eye(4)
    result[:3, :3] = [
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ]
    result[:3, 3] = t
    return result


def inverse_rigid(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def sensor_to_reference(sensor_calibration: dict, sensor_pose: dict, reference_pose: dict) -> np.ndarray:
    def matrix(record):
        return rigid_transform(record["translation"], record["rotation"])
    return inverse_rigid(matrix(reference_pose)) @ matrix(sensor_pose) @ matrix(sensor_calibration)


def project_to_image(points_reference, reference_from_camera, intrinsic):
    camera = transform_points(points_reference, inverse_rigid(reference_from_camera))
    depth = camera[:, 2]
    pixels_h = camera @ np.asarray(intrinsic).T
    pixels = pixels_h[:, :2] / np.maximum(depth[:, None], 1e-9)
    return pixels, depth

