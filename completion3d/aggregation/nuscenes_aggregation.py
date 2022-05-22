import os
import sys
import pickle
import zlib
import numpy as np

from PIL import Image
from scipy.spatial import cKDTree as KDTree
from scipy.interpolate import BPoly
from tqdm.autonotebook import tqdm

from nuscenes.nuscenes import NuScenes
from ..utils.transformations import (
    transform3d, interpolate,
    transformation3d_from_rotation_translation,
    transformation3d_from_quaternion_translation
)
from .common import (
    extract_objects,
    reconstruct_object,
    reconstruct_scene
)

_sensors = [
    'LIDAR_TOP',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_FRONT_LEFT'
]
_cameras = _sensors[1:]


def convert_nuscenes_boxes(boxes):
    locs = np.array([b.center for b in boxes]).reshape(-1, 3)
    dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
    rots = np.array([b.orientation.yaw_pitch_roll[0]
                        for b in boxes]).reshape(-1, 1)
    return np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
    

def points_rgb_from_image(points, image, lidar2cam=np.identity(4)):
    image = np.swapaxes(image, 0, 1)

    # 1. Transform points to camera frame
    points_cam = transform3d(points, lidar2cam)
    points_cam[:,:2] = points_cam[:,:2] / points_cam[:,2:3]

    # 2. Filter out-of-bound points
    valid_mask = points_cam[:,2] > 0
    valid_mask &= points_cam[:,0] >= 0
    valid_mask &= points_cam[:,0] <= image.shape[0] - 1
    valid_mask &= points_cam[:,1] >= 0
    valid_mask &= points_cam[:,1] <= image.shape[1] - 1

    # 3. Bilinear interpolate colors at projected points
    points_rgb = np.zeros((points.shape[0], 3))
    points_rgb[valid_mask] = interpolate(image, points_cam[valid_mask][:,:2])

    return points_rgb, valid_mask


def get_nuscenes_aggregation_infos( nusc: NuScenes ) -> dict:
    """Returns the aggregation information in the following format:
    ```
        {
            'frame_id': {
                'scene_id': the sequence corresponding to the current frame,
                'object_ids': a list of ids corresponding to each object,
                'global2lidar': a 4x4 transformation from global coordinate
                system to lidar coordinate system
            },
            ...
        }
    ```
    :param nusc: the nuscenes dataset object
    :type nusc: NuScenes
    :returns: the aggregation information
    :rtype: dict
    """

    with open(f'{nusc.dataroot}/nuscenes_infos_train.pkl', 'rb') as f:
        infos = {info['token']: info for info in pickle.load(f)['infos']}
    with open(f'{nusc.dataroot}/nuscenes_infos_val.pkl', 'rb') as f:
        infos.update({info['token']: info for info in pickle.load(f)['infos']})

    agginfos = {}
    for scene in nusc.scene:
        token = scene['first_sample_token']
        while token != '':
            s = nusc.get('sample', token)
            sd = nusc.get('sample_data', s['data']['LIDAR_TOP'])
            sensor2ego = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
            ego2global = nusc.get('ego_pose', sd['ego_pose_token'])
            lidar2global = transformation3d_from_quaternion_translation(
                np.roll(ego2global['rotation'], -1),
                ego2global['translation']
            ) @ transformation3d_from_quaternion_translation(
                np.roll(sensor2ego['rotation'], -1),
                sensor2ego['translation']
            )
            global2lidar = np.linalg.inv(lidar2global)

            agginfos[token] = {}
            agginfos[token]['global2lidar'] = global2lidar
            agginfos[token]['scene_id'] = scene['token']
            agginfos[token]['object_ids'] = []
            for ann_token in s['anns']:
                ann = nusc.get('sample_annotation', ann_token )
                agginfos[token]['object_ids'].append(ann['instance_token'])
            token = s['next']

    return agginfos


def aggregate_nuscenes_sequence( nusc, scene_idx, output_path=None ):
    ############################################################################
    # Initialize aggregation
    ############################################################################
    scene = nusc.scene[scene_idx]

    # Previous/current sample and next sample
    s_prev = nusc.get('sample', scene['first_sample_token'])
    s_next = None
    points_prev = None

    # sd_token: sample data token
    sd_token = {sensor: s_prev['data'][sensor] for sensor in _sensors}

    # sd: sample data, sensor -> sensor data
    sd = {}

    # sensor2global: sensor -> global transformation matrices
    sensor2global = {}
    lidar2global_list =[]

    # global2sensor: global -> sensor transformation matrices
    global2sensor = {}

    # Camera-related data
    cam_ts = {}
    cam_intrinsic = {}
    cam_load_next_frame = {}   # flag if we should load the next image
    image = {}

    # Aggregation results
    obj_points = {}
    obj_boxes = {}
    bg_points = []

    ############################################################################
    # Loop over all frames
    # ( including samples (annotated) and sweeps (unlabeled) )
    ############################################################################
    frame_pbar = tqdm(total=scene['nbr_samples'], desc=f'[{scene_idx}] Gathering sequence', leave=False)
    while sd_token['LIDAR_TOP'] != '':
        sd['LIDAR_TOP'] = nusc.get('sample_data', sd_token['LIDAR_TOP'])

        ########################################################################
        # Check if the camera frame needs to tbe updated
        ########################################################################
        for cam in _cameras:
            if cam not in cam_load_next_frame or sd[cam]['next'] == '':
                continue
            cam_sd_next = nusc.get('sample_data', sd[cam]['next'])
            cam_ts_next = cam_sd_next['timestamp'] / 1e6

            current_time_diff = np.abs(sd['LIDAR_TOP']['timestamp'] / 1e6 - cam_ts[cam])
            next_time_diff = np.abs(sd['LIDAR_TOP']['timestamp'] / 1e6 - cam_ts_next)
            if next_time_diff < current_time_diff:
                cam_load_next_frame[cam] = True
                sd_token[cam] = sd[cam]['next']
                if cam_sd_next['next'] != '':
                    cam_sd_nnext = nusc.get('sample_data', cam_sd_next['next'])
                    cam_ts_nnext = cam_sd_nnext['timestamp'] / 1e6
                    nnext_time_diff = np.abs(sd['LIDAR_TOP']['timestamp'] / 1e6 - cam_ts_nnext)
                    if nnext_time_diff < next_time_diff:
                        sd_token[cam] = cam_sd_next['next']

        ########################################################################
        # Get sample data and transformation matrices for each sensor
        ########################################################################
        for sensor in _sensors:
            if sensor in cam_load_next_frame and not cam_load_next_frame[sensor]:
                continue

            sd[sensor] = nusc.get('sample_data', sd_token[sensor])
            sensor2ego = nusc.get('calibrated_sensor', sd[sensor]['calibrated_sensor_token'])
            ego2global = nusc.get('ego_pose', sd[sensor]['ego_pose_token'])
            sensor2global[sensor] = transformation3d_from_quaternion_translation(
                np.roll(ego2global['rotation'], -1),
                ego2global['translation']
            ) @ transformation3d_from_quaternion_translation(
                np.roll(sensor2ego['rotation'], -1),
                sensor2ego['translation']
            )
            global2sensor[sensor] = np.linalg.inv(sensor2global[sensor])

            if 'CAM' in sensor:
                cam_intrinsic[sensor] = sensor2ego['camera_intrinsic']
        
        lidar2global_list.append(sensor2global['LIDAR_TOP'])

        ########################################################################
        # GT boxes for current frame
        # * For key frames: use existing box annotations
        # * For sweeps: interpolate boxes between two key frames
        ########################################################################
        if sd['LIDAR_TOP']['is_key_frame']:
            ####################################################################
            # Get current key frame's boxes, velocity vectors, and timestamps
            ####################################################################
            if s_next is None:
                # If this is the first key frame in the sequence (i.e., s_next
                # is None), get the annotations
                s_prev = nusc.get('sample', sd['LIDAR_TOP']['sample_token'])
                boxes_prev, velos_prev, ts_prev = {}, {}, {}

                # For each annotation, record box, velocity, and timestamp
                box_list = convert_nuscenes_boxes(nusc.get_sample_data(s_prev['data']['LIDAR_TOP'])[1])
                for ann_token, box in zip(s_prev['anns'], box_list): 
                    ann = nusc.get('sample_annotation', ann_token)
                    instance_token = ann['instance_token']

                    # Replace box xyz with global coordinate to compute the the
                    # interpolation later
                    box[:3] = ann['translation']
                    boxes_prev[instance_token] = box

                    # We have the gradient for xyz components (i.e., velocity).
                    # For other components (size, orientation), use 0 gradient
                    velos_prev[instance_token] = np.zeros(box.shape)
                    velos_prev[instance_token][:3] = nusc.box_velocity(ann_token)

                    # Timestamp needs to be recorded in seconds
                    ts_prev[instance_token] = s_prev['timestamp'] / 1e6
            else:
                # If the next key frame is defined, then reuse it
                # Previous interval:  s_prev ------ s_next
                #                                      v
                # Current interval:                 s_prev ------ s_next
                s_prev = s_next
                boxes_prev = boxes_next
                velos_prev = velos_next
                ts_prev = ts_next
            boxes = boxes_prev

            ####################################################################
            # Get next key frame's boxes, velocity vectors, and timestamps
            ####################################################################
            if s_prev['next'] != '':
                s_next = nusc.get('sample', s_prev['next'])
                boxes_next, velos_next, ts_next = {}, {}, {}
                box_list = convert_nuscenes_boxes(nusc.get_sample_data(s_next['data']['LIDAR_TOP'])[1])
                for ann_token, box in zip(s_next['anns'], box_list):
                    ann = nusc.get('sample_annotation', ann_token)
                    instance_token = ann['instance_token']
                    box[:3] = ann['translation']
                    boxes_next[instance_token] = box
                    velos_next[instance_token] = np.zeros(box.shape)
                    velos_next[instance_token][:3] = nusc.box_velocity(ann_token)
                    ts_next[instance_token] = s_next['timestamp'] / 1e6
            else:
                # If there's no next key frame, clear the data
                s_next = None
                boxes_next = {}
                velos_next = {}
                ts_next = {}
            
            ####################################################################
            # For each object annotated in both key frames, compute and save the
            # spline interpolator with velocity constraint if possible.
            ####################################################################
            interps = {}
            for instance_token in boxes_prev:
                if instance_token not in boxes_next:
                    continue
                
                # Check if velocity vector is valid
                if np.isnan(velos_prev[instance_token]).any():
                    grads_prev = [ boxes_prev[instance_token] ]
                else:
                    grads_prev = [ boxes_prev[instance_token], velos_prev[instance_token] ]
                if np.isnan(velos_next[instance_token]).any():
                    grads_next = [ boxes_next[instance_token] ]
                else:
                    grads_next = [ boxes_next[instance_token], velos_next[instance_token] ]

                # Compute a cubic Bezier curve interpolator
                # This has the same effect as Hermite interpolation
                # but uses Berstein basis instaed of Hermite functions.
                interps[instance_token] = BPoly.from_derivatives(
                    [ts_prev[instance_token], ts_next[instance_token]],
                    [grads_prev, grads_next]
                )
        else:
            if len(boxes_next) == 0:
                # Nothing to interpolate, discard the remaining unlabeled frames
                break

            boxes= {}
            for instance_token in boxes_prev:
                if instance_token not in boxes_next:
                    # The object has disappeared in the next key frame, so cannot interpolate
                    continue
                # Interpolate boxes using pre-computed interpolations
                boxes[instance_token] = interps[instance_token](sd['LIDAR_TOP']['timestamp'] / 1e6)

        ########################################################################
        # Translate box locations from global frame to LiDAR frame
        # (since the interpolation has to be done in the global coordinate)
        ########################################################################
        for instance_token, box in boxes.items():
            box[:3] = transform3d(box[np.newaxis,:3], global2sensor['LIDAR_TOP'])[0]
            if instance_token not in obj_boxes:
                obj_boxes[instance_token] = []
            obj_boxes[instance_token].append(box)

        ######################################################################## 
        # Load frame point cloud
        ######################################################################## 
        points = np.fromfile(f'{nusc.dataroot}/{sd["LIDAR_TOP"]["filename"]}', dtype=np.float32).reshape(-1, 5)

        ######################################################################## 
        # Extract image RGB features
        ######################################################################## 
        points_rgb = np.zeros((points.shape[0], 3))
        valid_rgb_mask = np.zeros(points.shape[0], dtype=bool)
        num_valid_rgb = np.zeros(points.shape[0])
        for cam in _cameras:
            # Load new camera frame if necessary
            if cam not in cam_load_next_frame or cam_load_next_frame[cam]:
                cam_ts[cam] = sd[cam]['timestamp'] / 1e6
                image[cam] = np.array(Image.open(f'{nusc.dataroot}/{sd[cam]["filename"]}'))

                cam_load_next_frame[cam] = False

                time_diff = sd["LIDAR_TOP"]["timestamp"]/1e6-cam_ts[cam]
                if time_diff > 0.05:
                   tqdm.write(f'Warning: large time difference for {cam}: {time_diff}')

            # Extract rgb features from image
            # https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py#L834
            lidar2cam = transformation3d_from_rotation_translation(
                    cam_intrinsic[cam]
                ) @ global2sensor[cam] @ sensor2global['LIDAR_TOP']
            points_rgb_i, valid_rgb_mask_i = points_rgb_from_image(points, image[cam], lidar2cam)

            # Record rgb features and number of images the points are observed in
            points_rgb += points_rgb_i
            valid_rgb_mask |= valid_rgb_mask_i
            num_valid_rgb[valid_rgb_mask_i] += 1

        # Add camera RGB colors to point cloud
        points_rgb[valid_rgb_mask] = points_rgb[valid_rgb_mask] / num_valid_rgb[valid_rgb_mask][:,np.newaxis]
        points_rgb = np.clip(points_rgb, 0, 255)

        ######################################################################## 
        # Extract semantic labels for each point
        ######################################################################## 
        if sd['LIDAR_TOP']['is_key_frame']:
            lidarseg = nusc.get('lidarseg', sd['LIDAR_TOP']['token'])
            points_labels = np.fromfile(f'{nusc.dataroot}/{lidarseg["filename"]}', dtype=np.uint8)[:,np.newaxis]
        else:
            # If the current frame doesn't have segmentation labels,
            # then we assign the label based on nearest points
            points_labels = np.zeros((points.shape[0], 1))
            _, ind = KDTree(points_prev[:,:3]).query(
                transform3d(points[:,:3], sensor2global['LIDAR_TOP']),
                k=1, distance_upper_bound=1, n_jobs=-1
            )
            valid = ind < points_prev.shape[0]
            points_labels[valid,0] = points_prev[ind[valid],-1]

        points = np.concatenate([points, points_rgb, points_labels], axis=-1)
        points_prev = points.copy()
        points_prev[:,:3] = transform3d(points_prev[:,:3], sensor2global['LIDAR_TOP'])

        # Remove noisy points and points on ego vehicle
        points = points[(points[:,-1] >= 1) & (points[:,-1] <= 30)]

        ######################################################################## 
        # Extract object and background point clouds
        ######################################################################## 
        fg_mask = (points[:,-1] >= 2) & (points[:,-1] <= 23)
        obj_points, obj_mask = extract_objects(
            points[fg_mask], boxes, origin=(0.5, 0.5, 0.5), obj_points=obj_points
        )
        bg_points.append(points[points[:,-1] >= 24])

        ######################################################################## 
        # Go to the next frame
        ######################################################################## 
        sd_token['LIDAR_TOP'] = sd['LIDAR_TOP']['next']

        if sd['LIDAR_TOP']['is_key_frame']:
            # Update the progress bar
            frame_pbar.update()
    frame_pbar.close()

    ############################################################################
    # Concatenate (and align) point clouds
    ############################################################################
    for instance_token in tqdm(obj_points, total=len(obj_points), desc=f'[{scene_idx}] Reconstructing objects', leave=False):
        obj_points[instance_token] = reconstruct_object(
            obj_points[instance_token], obj_boxes[instance_token], origin=(0.5, 0.5, 0.5),
            mirror=True, remove_ground=False, icp_params=None
        )
    for _ in tqdm(range(1), desc=f'[{scene_idx}] Reconstructing scene', leave=False):
        bg_points = reconstruct_scene(bg_points, lidar2global_list)

    ############################################################################
    # Save scene and object point clouds
    ############################################################################
    if output_path is not None:
        os.makedirs(f'{output_path}/scenes', exist_ok=True)
        os.makedirs(f'{output_path}/objects', exist_ok=True)
        with open(f'{output_path}/scenes/{scene["token"]}.bin', 'wb') as f:
            f.write(zlib.compress(bg_points.flatten().tobytes()))
        for instance_token, obj_points_i in obj_points.items():
            with open(f'{output_path}/objects/{instance_token}.bin', 'wb') as f:
                f.write(zlib.compress(obj_points_i.flatten().tobytes()))
    else:
        return bg_points, obj_points


def aggregate_nuscenes( nusc, begin=0, end=850, output_path=None ):
    for scene_idx in tqdm(range(begin, end), desc=f'NuScenes'):
        aggregate_nuscenes_sequence( nusc, scene_idx, output_path )


if __name__ == '__main__':
    if len(sys.argv) == 2:
        scene_begin = 0
        scene_end = 850
    elif len(sys.argv) == 4:
        scene_begin = int(sys.argv[2])
        scene_end = int(sys.argv[3])
    else:
        raise ValueError(f'Usage: {sys.argv[0]} <dataset_path> [idx_begin] [idx_end]')

    dataset_path = sys.argv[1]
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(dataset_path)

    output_path = f'{dataset_path}/nuscenes_agg'
    os.makedirs(output_path, exist_ok=True)

    nusc = NuScenes(version='v1.0-trainval', dataroot=dataset_path)

    agginfos_path = f'{output_path}/nuscenes_agginfos_trainval.pkl'
    if not os.path.isfile(agginfos_path):
        agginfos = get_nuscenes_aggregation_infos(nusc)
        with open(agginfos_path, 'wb') as f:
            pickle.dump(agginfos, f)
        del agginfos

    aggregate_nuscenes( nusc, begin=scene_begin, end=scene_end, output_path=output_path )
