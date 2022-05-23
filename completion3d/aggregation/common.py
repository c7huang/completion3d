import zlib
import numpy as np
from typing import List, Optional
try:
    from numpy.typing import ArrayLike
except:
    from typing import Union
    ArrayLike = Union[tuple, list, np.ndarray]
from sklearn.linear_model import RANSACRegressor
from mmdet3d.core.bbox.box_np_ops import points_in_rbbox
from ..utils.transformations import (
    rotate2d, transform3d, interpolate
)
from ..utils.o3dutils import (
    incremental_icp, 
    voxel_grid_downsample,
    statistical_outlier_removal,
    estimate_normals,
)


def ransac_ground_segmentation(points, size, origin=(0.5, 0.5, 0.0)):
    '''
    A simple ground segmentation method based on RANSAC plane fitting.
    This method is designed to be applied to aggregated point clouds for
    single objects, and assumes the ground is a plane with slope < 10 degrees.
    '''

    l = max(size[0], size[1])
    ground_mask = points[:,2] < -origin[2]*size[2] + l*0.0875    # tan(5)
    candidates = points[ground_mask]
    if candidates.shape[0] < 100:
        return np.zeros(points.shape[0], dtype=bool)

    reg = RANSACRegressor(
        residual_threshold=0.05,
        max_trials=1e4,
        stop_probability=1.0,
        random_state=42
    )
    reg.fit(candidates[:,:2], candidates[:,2])
    ground_mask[ground_mask] = reg.inlier_mask_
    return ground_mask


def extract_objects(points, boxes, origin=(0.5, 0.5, 0.0), obj_points=None):
    if obj_points is None:
        obj_points = {}

    if len(boxes) == 0:
        return obj_points, np.zeros((points.shape[0], 1), dtype=bool)

    # Identify points in boxes
    # obj_mask: (# points, # boxes)
    obj_mask = points_in_rbbox(points, np.stack(list(boxes.values())), origin=origin)

    for i, (id, box) in enumerate(boxes.items()):
        obj_points_i = points[obj_mask[:,i]]
        if id in obj_points:
            if not isinstance(obj_points[id], list):
                obj_points[id] = [obj_points[id]]
            obj_points[id].append(obj_points_i)
        else:
            obj_points[id] = obj_points_i

    return obj_points, obj_mask


def reconstruct_object(
    points_list: List[ArrayLike], box_list: ArrayLike, origin: ArrayLike = (0.5, 0.5, 0.0), 
    mirror: bool = True, remove_ground: bool = True, icp_params: Optional[dict] = None
) -> np.ndarray:
    '''
    Aggregate a sequence of objects based on the approach proposed in:
        Manivasagam, Sivabalan, et al. "Lidarsim: Realistic
        lidar simulation by leveraging the real world." CVPR. 2020.
    The proposed approach is not reimplemented exactly due to lack of details.

    Steps implemented in this method:
        0. Object points + features extraction
           (implemented in :func:`completion3d.aggregation.common.extract_objects`)
        1. Alignment using ground truth annotations
        2. Mirror the object along the heading axis
        3. Refine alignment using ICP (iterative colored-ICP)
        4. Ground point removal (ransac plane fitting)
        5. Voxel-grid down sample with voxel size of (2cm x 2cm x 2cm)
        6. Statistical outlier removal
        7. Normal estimation
        8. Incidence angle calculation
    
    :param points_list: a list of N point clouds representing a sequence of
        observations for a particular object. The point clouds are expected to
        be unaligned in the original lidar frame.
        Format: (x,y,z,lidar,r,g,b,label)
    :type points_list: List[array_like]
    :param box_list: a list of bounding boxes (N, 7) in lidar frame
    :type box_list: array_like
    :param origin: a 3-tuple indicating the location of the xyz encoding inside
        the bounding box 
    :type origin: ArrayLike
    :param mirror: if the object should be mirrored along its jheading axis
    :type mirror: bool
    :param remove_ground: if the ground points should be removed
    :type remove_ground: bool
    :param icp_params: parameters for additional ICP alignment. Set it to None
        to disable ICP alignment.
    :type icp_params: dict or None
    :returns: a 2-dimensional array containing the reconstructed points. Each
        point is encoded in the following format:
        (x,y,z,n_x,n_y,n_z,r,g,b,label,range,incidence,lidar)
    :rtype: np.ndarray
    '''

    if not isinstance(points_list, list):
        points_list = [points_list]

    if np.concatenate(points_list).shape[0] == 0:
        return np.zeros((0, points_list[0].shape[1]+5), dtype=np.float32)

    for i, (points, box) in enumerate(zip(points_list, box_list)):
        # Extract lidar ray direction vector (for computing incidence angle)
        # (x,y,z,lidar,r,g,b,label,range,ray_x,ray_y,ray_z)
        ranges = np.linalg.norm(points[:,:3], axis=-1)[:,np.newaxis]
        rays = points[:,:3] / ranges
        points = np.concatenate([points, ranges, rays], axis=-1)

        # 1. Alignment using ground truth annotations
        # Note: also requires rotation of ray vectors
        points[:,:3] -= box[:3]
        points[:,:2] = rotate2d(points[:,:2], box[6])
        points[:,-3:-1] = rotate2d(points[:,-3:-1], box[6])

        # 2. Mirror the object along the heading axis
        if mirror:
            points_sym = points.copy()
            points_sym[:,[0,-3]] *= -1
            points_list[i] = np.concatenate([points, points_sym])
        else:
            points_list[i] = points

    # 3. Refine alignment using ICP (iterative colored-ICP)
    if icp_params is not None:
        intensities = np.concatenate([points_i[:,3] for points_i in points_list])
        intensity_min = np.log(intensities.min()+1)
        intensity_max = np.log(intensities.max()+1)
        intensity_range = intensity_max - intensity_min

        points_list_icp = []
        for points_i in points_list:
            points_i = points_i[:,:4].copy()
            points_i[:,3] = (np.log(points_i[:,3]+1) - intensity_min) / intensity_range
            points_list_icp.append(points_i)
        
        # Compute ICP transformations
        transformations = incremental_icp(points_list_icp, **icp_params)

        for i in range(len(transformations)):
            points_list[i][:,:3] = transform3d(points_list[i][:,:3], transformations[i])
            # Apply any rotations to ray directions as well
            points_list[i][:,-3:] = np.dot(points_list[i][:,-3:], transformations[i][:3,:3].T)

    # Concatenate points
    points = np.concatenate(points_list)

    if points.shape[0] > 100:
        # 4. Ground point removal (ransac plane fitting)
        if remove_ground:
            ground_mask = ransac_ground_segmentation(points, np.max(box_list, axis=0)[3:6], origin)
            points = points[~ground_mask]

        # 5. Voxel-grid down sample with voxel size of (2cm x 2cm x 2cm)
        _, ind = voxel_grid_downsample(points[:,:3], 0.02, average=False)
        points = points[ind]

        # 6. Statistical outlier removal
        points = statistical_outlier_removal(points, 30, 2.0)

        # 7. Normal estimation
        # (x,y,z,lidar,r,g,b,label,range,ray_x,ray_y,ray_z,n_x,n_y,n_z)
        normals = estimate_normals(points, radius=0.1, k=200).astype(np.float32)
        normals = np.clip(normals, -1, 1)
        points = np.concatenate([points, normals], axis=-1)

        # 8. Incidence angle calculation
        # (x,y,z,lidar,r,g,b,label,range,ray_x,ray_y,ray_z,n_x,n_y,n_z,incidence)
        rays = points[:,-6:-3]
        normals = points[:,-3:]
        incidence = np.arccos(normals[:,0]*rays[:,0]+normals[:,1]*rays[:,1]+normals[:,2]*rays[:,2])
        flip_mask = incidence > np.pi/2
        points[flip_mask,-3:] *= -1
        incidence[flip_mask] = np.pi - incidence[flip_mask]
        points = np.concatenate([points, incidence[:,np.newaxis].astype(np.float32)], axis=-1)    
    else:
        points = np.concatenate([points, -points[:,-3:], np.zeros((points.shape[0], 1), dtype=np.float32)], axis=-1)

    # Reformat and drop ray information
    #    (x,y,z,lidar,r,g,b,label,range,ray_x,ray_y,ray_z,n_x,n_y,n_z,incidence)
    # -> (x,y,z,n_x,n_y,n_z,r,g,b,label,range,incidence,lidar)
    num_lidar_features = points.shape[-1] - 15
    points = np.concatenate([
        points[:,[0,1,2,-4,-3,-2,-12,-11,-10,-9,-8,-1]],
        points[:,3:3+num_lidar_features]
    ], axis=-1)

    return points


def reconstruct_scene(points_list: List[ArrayLike], lidar2global_list: ArrayLike) -> np.ndarray:
    # See Section 3.1 in Manivasagam, Sivabalan, et al. "Lidarsim: Realistic
    # lidar simulation by leveraging the real world." CVPR. 2020.

    for i, (points, lidar2global) in enumerate(zip(points_list, lidar2global_list)):
        # 0. Extract lidar ray direction vector (for computing incidence angle)
        # (x,y,z,lidar,r,g,b,label,range,ray_x,ray_y,ray_z)
        ranges = np.linalg.norm(points[:,:3], axis=-1)[:,np.newaxis]
        rays = points[:,:3] / ranges
        points = np.concatenate([points, ranges, rays], axis=-1)

        # 1. Initial alignment with SLAM
        points[:,:3] = transform3d(points[:,:3], lidar2global)
        points[:,-3:] = np.dot(points[:,-3:], lidar2global[:3,:3].T)
        points_list[i] = points

    # 2. Aggregate aligned background points
    points = np.concatenate(points_list)

    # 3. Voxel-grid down sample with voxel size of (4cm x 4cm x 4cm)
    _, ind = voxel_grid_downsample(points[:,:3], 0.04, average=False)
    points = points[ind]

    # 4. Statistical outlier removal
    points = statistical_outlier_removal(points, 30, 2.0)

    # 5. Normal estimation
    # (x,y,z,lidar,r,g,b,label,range,ray_x,ray_y,ray_z,n_x,n_y,n_z)
    normals = estimate_normals(points, radius=0.2, k=200).astype(np.float32)
    normals = np.clip(normals, -1, 1)
    points = np.concatenate([points, normals], axis=-1)

    # 6. Incidence angle calculation
    # (x,y,z,lidar,r,g,b,label,range,ray_x,ray_y,ray_z,n_x,n_y,n_z,incidence)
    rays = points[:,-6:-3]
    normals = points[:,-3:]
    incidence = np.arccos(normals[:,0]*rays[:,0]+normals[:,1]*rays[:,1]+normals[:,2]*rays[:,2])
    flip_mask = incidence > np.pi/2
    points[flip_mask,-3:] *= -1
    incidence[flip_mask] = np.pi - incidence[flip_mask]
    points = np.concatenate([points, incidence[:,np.newaxis].astype(np.float32)], axis=-1)    

    # Reformat and drop ray information
    #    (x,y,z,lidar,r,g,b,label,range,ray_x,ray_y,ray_z,n_x,n_y,n_z,incidence)
    # -> (x,y,z,n_x,n_y,n_z,r,g,b,label,range,incidence,lidar)
    num_lidar_features = points.shape[-1] - 15
    points = np.concatenate([
        points[:,[0,1,2,-4,-3,-2,-12,-11,-10,-9,-8,-1]],
        points[:,3:3+num_lidar_features]
    ], axis=-1)

    return points


def load_aggregated_points(
    agg_dataset_path, scene_id, object_ids, gt_boxes,
    global2lidar=np.identity(4), num_features=14
):
    # Load scene point cloud
    with open(f'{agg_dataset_path}/scenes/{scene_id}.bin', 'rb') as f:
        bg_points = np.array(np.frombuffer(zlib.decompress(f.read()), dtype=np.float32).reshape(-1, num_features))
        # Transform background points
        bg_points[:,:3] = transform3d(bg_points[:,:3], global2lidar)
        bg_points[:,3:6] = transform3d(bg_points[:,3:6], np.linalg.inv(global2lidar).T)

    # Load object point clouds
    obj_points = {}
    for obj_id, box in zip(object_ids, gt_boxes):
        with open(f'{agg_dataset_path}/objects/{obj_id}.bin', 'rb') as f:
            obj_points[obj_id] = np.array(np.frombuffer(zlib.decompress(f.read()), dtype=np.float32).reshape(-1, num_features))
            obj_points[obj_id][:,:2] = rotate2d(obj_points[obj_id][:,:2], -box[6])
            obj_points[obj_id][:,3:5] = rotate2d(obj_points[obj_id][:,3:5], -box[6])
            obj_points[obj_id][:,:3] += box[:3]

    return bg_points, obj_points
