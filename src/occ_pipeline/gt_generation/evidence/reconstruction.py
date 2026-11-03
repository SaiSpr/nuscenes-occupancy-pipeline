"""Optional static Poisson surface proposal. Never changes measured supervision.

Open3D reconstructs a surface shell, not verified occupancy or free space.
Normals face the corresponding acquisition origin. Trim weakly supported mesh
vertices and reject proposals too far from actual static returns.
"""

import numpy as np


def poisson_candidates(sweeps, grid, config):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ValueError("Poisson requested but Open3D is missing. Install the [reconstruction] extra or use the Docker reconstruction target.") from exc
    groups, origins = [], []
    for sweep in sweeps:
        p = sweep['points'][sweep['static']]
        p = p[grid.contains(p) & (np.linalg.norm(p-sweep['origin'], axis=1) <= config.max_range_m)]
        groups.append(p)
        origins.append(np.broadcast_to(sweep['origin'], p.shape))
    points = np.concatenate(groups) if groups else np.empty((0,3))
    views = np.concatenate(origins) if origins else np.empty((0,3))
    # Deterministic cap; no randomized sampling or stochastic validation masks.
    if len(points) > config.poisson_max_points:
        indices = np.linspace(0, len(points)-1, config.poisson_max_points, dtype=int)
        points, views = points[indices], views[indices]
    mask = np.zeros(grid.shape, dtype=bool)
    stats = {'backend': 'open3d', 'version': o3d.__version__, 'input_static_points': len(points),
             'status': 'insufficient_points', 'surface_proposals': 0}
    if len(points) < 100:
        return mask, stats
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=grid.voxel_size*3, max_nn=30))
    normals = np.asarray(cloud.normals).copy()
    flip = (normals*(views-points)).sum(axis=1) < 0
    normals[flip] *= -1
    cloud.normals = o3d.utility.Vector3dVector(normals)
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=config.poisson_depth, n_threads=1)
    density = np.asarray(density)
    if not len(density):
        stats['status'] = 'empty_mesh'
        return mask, stats
    mesh.remove_vertices_by_mask(density < np.quantile(density, config.poisson_trim_quantile))
    mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(np.array(grid.minimum), np.array(grid.maximum)))
    voxels = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh, grid.voxel_size, np.array(grid.minimum), np.array(grid.maximum))
    centers = np.asarray([voxels.get_voxel_center_coordinate(v.grid_index) for v in voxels.get_voxels()]).reshape(-1,3)
    centers = centers[grid.contains(centers)]
    if len(centers):
        candidates = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centers))
        supported = np.asarray(candidates.compute_point_cloud_distance(cloud)) <= config.poisson_support_m
        indices = grid.indices(centers[supported])
        mask[tuple(indices.T)] = True
    stats.update(status='completed', mesh_vertices=len(mesh.vertices), mesh_triangles=len(mesh.triangles),
                 surface_proposals=int(mask.sum()))
    return mask, stats
