import zlib
import numpy as np
from glob import glob
from typing import List, Optional
try:
    from numpy.typing import ArrayLike
except:
    from typing import Union
    ArrayLike = Union[tuple, list, np.ndarray]
from sklearn.linear_model import RANSACRegressor
from ..utils.box_np_ops import points_in_rbbox
from ..utils.transforms import (
    rotate2d, transform3d_affine,
    transformation3d_from_euler
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
            points_list[i][:,:3] = transform3d_affine(points_list[i][:,:3], transformations[i])
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
        points[:,:3] = transform3d_affine(points[:,:3], lidar2global)
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


def filter_points_by_range(points, point_cloud_range=None):
    if point_cloud_range is None:
        return points
    else:
        range_mask = points[:,0] > point_cloud_range[0]
        range_mask &= points[:,0] < point_cloud_range[3]
        range_mask &= points[:,1] > point_cloud_range[1]
        range_mask &= points[:,1] < point_cloud_range[4]
        range_mask &= points[:,2] > point_cloud_range[2]
        range_mask &= points[:,2] < point_cloud_range[5]
        return points[range_mask]


def load_aggregated_points(
    scene_path: str, object_path: str,
    scene_id: str, object_ids: ArrayLike,
    scene_transform: ArrayLike, object_transforms: ArrayLike,
    num_point_features: int,
    use_point_features: Union[ArrayLike, int, None] = None,
    scene_chunks: Optional[ArrayLike] = None,
    global_transform: Optional[ArrayLike] = None,
    point_cloud_range: Optional[ArrayLike] = None,
    compression: Union[bool, str, None] = None,
    combine: bool = True
) -> Union[np.ndarray, tuple]:
    """Load aggregated point cloud

    :param scene_path: the location containing the aggregated scene point
        clouds. The point clouds are stored in bytes and can be loaded into
        float32 numpy arrays.
    :type scene_path: str
    :param object_path: the location containing the aggregated object point
        clouds. The point clouds are stored in bytes and can be loaded into
        float32 numpy arrays.
    :type object_path: str
    :param scene_id: the identifier for the scene
    :type scene_id: str
    :param object_ids: a list of N identifiers for the objects
    :type object_ids: ArrayLike
    :param scene_transform: a 4x4 transformation matrix that transforms the
        scene point cloud from global to lidar frame
    :type scene_transform: ArrayLike
    :param object_transforms: a (N,4) array representing the translation and
        rotation (yaw) for each object
    :type object_transforms: ArrayLike
    :param num_point_features: the number of features in the aggregated point
        clouds. This should be 14 for nuscenes and 15 for Waymo
    :type num_point_features: int
    :param use_point_features: the indices of the point features that will be
        kept and used after loading
    :type use_point_features: ArrayLike | int | None, optional
    :param scene_chunks: _description_, defaults to None
    :type scene_chunks: ArrayLike | None, optional
    :param global_transform: a global transformation that will be applied   
        to the aggregated point cloud before clipping by range
    :type global_transform: ArrayLike | None, optional
    :param point_cloud_range: the range of the point cloud in 
        `[x_min, y_min, z_min, x_max, y_max, z_max]` format.
        The points outside of this range will be discarded.
    :type point_cloud_range: ArrayLike | None, optional
    :param compression: if the aggregated dataset is in compressed format (e.g.,
        zlib), specify the compression algorithm here
    :type compression: bool | str | None, optional
    :param combine: if set to False, return bg points and object points
        separatedly in a tuple, defaults to True
    :type combine: bool, optional
    :return: the aggregated point clouds
    :rtype: np.ndarray | tuple
    """

    if isinstance(use_point_features, int):
        use_point_features = list(range(use_point_features))
    elif use_point_features is None or not isinstance(use_point_features, (list, tuple, np.ndarray)):
        use_point_features = list(range(num_point_features))
    transform_normals = 3 in use_point_features or 4 in use_point_features or 5 in use_point_features

    object_transforms = transformation3d_from_euler(
        'z', -object_transforms[:,3], object_transforms[:,:3]
    )
    if global_transform is not None:
        scene_transform = global_transform @ scene_transform
        object_transforms = global_transform @ object_transforms

    # Load scene point cloud
    bg_points = []
    if scene_chunks is None:
        scene_chunks = range(len(glob(f'{scene_path}/{scene_id}.bin.*')))
    for c in scene_chunks:
        with open(f'{scene_path}/{scene_id}.bin.{c}', 'rb') as f:
            if compression is None or compression is False:
                data = f.read()
            elif compression == 'zlib':
                data = zlib.decompress(f.read())
            else:
                raise ValueError(f'unsupported compression algorithm: {compression}')
        bg_points.append(
            np.frombuffer(data, dtype=np.float32).reshape(-1, num_point_features)
        )
    bg_points = np.concatenate(bg_points)
    # Transform background points
    bg_points[:,:3] = transform3d_affine(bg_points[:,:3], scene_transform)
    if transform_normals:
        bg_points[:,3:6] = transform3d_affine(bg_points[:,3:6], np.linalg.inv(scene_transform).T)
    if not combine:
        bg_points = filter_points_by_range(bg_points, point_cloud_range)[:,use_point_features]

    # Load object point clouds
    obj_points = {}
    for obj_id, obj_transform in zip(object_ids, object_transforms):
        with open(f'{object_path}/{obj_id}.bin', 'rb') as f:
            if compression is None or compression is False:
                data = f.read()
            elif compression == 'zlib':
                data = zlib.decompress(f.read())
            else:
                raise ValueError(f'unsupported compression algorithm: {compression}')
        obj_points_i = np.frombuffer(data, dtype=np.float32).reshape(-1, num_point_features).copy()
        obj_points_i[:,:3] = transform3d_affine(obj_points_i[:,:3], obj_transform)
        if transform_normals:
            obj_points_i[:,3:6] = transform3d_affine(obj_points_i[:,3:6], np.linalg.inv(obj_transform).T)
        if not combine:
            obj_points_i = filter_points_by_range(obj_points_i, point_cloud_range)[:,use_point_features]
        obj_points[obj_id] = obj_points_i

    if combine:
        return filter_points_by_range(
            np.concatenate([bg_points, *list(obj_points.values())]),
            point_cloud_range
        )[:,use_point_features]
    else:
        return bg_points, obj_points
