import os
import sys
import numpy as np
import pickle

from scipy.spatial.transform import Rotation as R
from tqdm.autonotebook import tqdm
from nuscenes.nuscenes import NuScenes

from mmcv.fileio import FileClient
from mmdet3d.core.bbox.box_np_ops import points_in_rbbox
from mmdet3d.datasets.pipelines.loading import LoadPointsFromFile
from utils import rot_matrix_2d, extract_object_point_clouds

file_client = FileClient(backend='disk')
_load_points_from_file = LoadPointsFromFile(
    coord_type='LIDAR',
    load_dim=5,
    use_dim=5,
    file_client_args=dict(backend='disk')
)


def load_points_from_file(filename):
    results = dict(pts_filename=filename)
    _load_points_from_file(results)
    return results['points'].tensor.numpy()


def get_nuscenes_infos( nusc ):
    with open(f'{nusc.dataroot}/nuscenes_infos_train.pkl', 'rb') as f:
        infos = {info['token']: info for info in pickle.load(f)['infos']}
    with open(f'{nusc.dataroot}/nuscenes_infos_val.pkl', 'rb') as f:
        infos.update({info['token']: info for info in pickle.load(f)['infos']})

    for scene in nusc.scene:
        token = scene['first_sample_token']
        while token != '':
            sample = nusc.get('sample', token)
            infos[token]['scene_token'] = scene['token']
            infos[token]['instance_tokens'] = []
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token )
                infos[token]['instance_tokens'].append(ann['instance_token'])
            token = sample['next']

    return infos


def aggregate_nuscenes_sequence( nusc, infos, scene_idx, output_path=None ):
    scene = nusc.scene[scene_idx]
    frame_pbar = tqdm(total=scene['nbr_samples'], desc=f'Scene {scene_idx}', leave=False)

    s_prev = nusc.get('sample', scene['first_sample_token'])
    s_next = None
    
    obj_points = {}
    bg_points = []
    
    token = s_prev['data']['LIDAR_TOP']
    while token != '':
        sd = nusc.get('sample_data', token)
        
        ######################################################################## 
        # GT boxes for current frame
        # * For key frames: use box annotations in MMDet3D format
        # * For sweeps: interpolate boxes between two key frames
        ######################################################################## 
        if sd['is_key_frame']:
            # Get current key frame (s_prev) boxes
            if s_next is None:
                s_prev = nusc.get('sample', sd['sample_token'])
                info_prev = infos[s_prev['token']]
                boxes_prev = {}
                for ann_token, box in zip(s_prev['anns'], info_prev['gt_boxes']):
                    instance_token = nusc.get('sample_annotation', ann_token)['instance_token']
                    boxes_prev[instance_token] = box
            else:
                s_prev = s_next
                info_prev = info_next
                boxes_prev = boxes_next
            boxes= boxes_prev

            # Get next key frame (s_next) boxes
            if s_prev['next'] != '':
                s_next = nusc.get('sample', s_prev['next'])
                info_next = infos[s_next['token']]
                boxes_next = {}
                for ann_token, box in zip(s_next['anns'], info_next['gt_boxes']):
                    instance_token = nusc.get('sample_annotation', ann_token)['instance_token']
                    boxes_next[instance_token] = box
            else:
                s_next = None
                info_next = None
                boxes_next = {}
            
            frame_pbar.update()
            
        else:
            if len(boxes_next) == 0:
                # Nothing to interpolate, discard the remaining unlabeled frames
                break
                
            # Compute interpolation factor base on timestamps
            lerp_t = (sd['timestamp'] - s_prev['timestamp']) / (s_next['timestamp'] - s_prev['timestamp'])
            boxes= {}
            for instance_token, box in boxes_prev.items():
                if instance_token not in boxes_next:
                    # The object has disappeared in the next key frame, so cannot interpolate
                    continue;
                boxes[instance_token] = (1-lerp_t) * boxes_prev[instance_token] + lerp_t * boxes_next[instance_token]

        ######################################################################## 
        # Load frame point cloud
        ######################################################################## 
        points = load_points_from_file(f'{nusc.dataroot}/{sd["filename"]}')
        points = np.concatenate([points, np.full((points.shape[0], 1), len(bg_points))], axis=1)

        # Remove noisy points near the LiDAR sensor
        points = points[np.linalg.norm(points[:,:3], axis=-1) > 2]

        ######################################################################## 
        # Extract and aggregate object point clouds
        ######################################################################## 
        obj_mask, obj_points = extract_object_point_clouds(
            points, boxes, origin=(0.5, 0.5, 0.5), 
            obj_points=obj_points, align=False
        )

        ######################################################################## 
        # Transform and aggregate background point clouds
        ######################################################################## 
        points = points[obj_mask.sum(-1) == 0]

        # Get calibration
        lidar2ego = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        lidar2ego_rotation = R.from_quat(np.roll(lidar2ego['rotation'], -1))
        lidar2ego_translation = np.array(lidar2ego['translation'])
        ego2global = nusc.get('ego_pose', sd['ego_pose_token'])
        ego2global_rotation = R.from_quat(np.roll(ego2global['rotation'], -1))
        ego2global_translation = np.array(ego2global['translation'])

        points[:,:3] = lidar2ego_rotation.apply(points[:,:3])
        points[:,:3] += lidar2ego_translation
        points[:,:3] = ego2global_rotation.apply(points[:,:3])
        points[:,:3] += ego2global_translation

        bg_points.append(points)

        # Go to next frame
        token = sd['next']
    

    # Concatenate point clouds
    bg_points = np.concatenate(bg_points)
    for token, obj_points_i in obj_points.items():
        obj_points[token] = np.concatenate(obj_points_i)
    
    ############################################################################
    # Save scene and object point clouds
    ############################################################################
    frame_pbar.close()

    if output_path is not None:
        os.makedirs(f'{output_path}/scenes', exist_ok=True)
        os.makedirs(f'{output_path}/instances', exist_ok=True)
        file_client.put(bg_points.flatten().tobytes(), f'{output_path}/scenes/{scene["token"]}.bin')
        for token, obj_points_i in obj_points.items():
            file_client.put(obj_points_i.flatten().tobytes(), f'{output_path}/instances/{token}.bin')
    else:
        return bg_points, obj_points


def aggregate_nuscenes( nusc, infos, begin=0, end=850, output_path=None ):
    for scene_idx in tqdm(range(begin, end), desc=f'NuScenes'):
        aggregate_nuscenes_sequence( nusc, infos, scene_idx, output_path )


def load_nuscenes_points_with_object_mask( dataset_path, info ):
    filename = info['lidar_path'].split('/')[-1]
    points = load_points_from_file(f'{dataset_path}/samples/LIDAR_TOP/{filename}')
    boxes = { token: box for token, box in zip(info['instance_tokens'], info['gt_boxes']) }
    obj_mask = points_in_rbbox(points, np.stack(list(boxes.values())), origin=(0.5, 0.5, 0.5))
    return points, obj_mask


def load_nuscenes_aggregated_points( dataset_path, info ):
    # Load scene point cloud
    bg_points = load_points_from_file(f'{dataset_path}/scenes/{info["scene_token"]}.bin')

    # Load object point clouds
    obj_points = {}
    for token in info['instance_tokens']:
        obj_points[token] = load_points_from_file(f'{dataset_path}/instances/{token}.bin')

    # Transform background points
    global2ego_translation = -np.array(info['ego2global_translation'])
    global2ego_rotation = R.from_quat(np.roll(info['ego2global_rotation'], -1)).inv()
    ego2lidar_translation = -np.array(info['lidar2ego_translation'])
    ego2lidar_rotation = R.from_quat(np.roll(info['lidar2ego_rotation'], -1)).inv()

    bg_points[:,:3] += global2ego_translation
    bg_points[:,:3] = global2ego_rotation.apply(bg_points[:,:3])
    bg_points[:,:3] += ego2lidar_translation
    bg_points[:,:3] = ego2lidar_rotation.apply(bg_points[:,:3])

    # Transform object points
    for token, box in zip( info['instance_tokens'], info['gt_boxes'] ):
        obj_points[token][:,:2] = np.dot(obj_points[token][:,:2], rot_matrix_2d(-box[6]).T)
        obj_points[token][:,:3] += box[:3]
        obj_points[token] = obj_points[token][
            (np.abs(obj_points[token][:,0]) < 54) & 
            (np.abs(obj_points[token][:,1]) < 54) &
            (obj_points[token][:,2] > -5) &
            (obj_points[token][:,2] < 3)
        ]

    bg_points = bg_points[
        (np.abs(bg_points[:,0]) < 54) & 
        (np.abs(bg_points[:,1]) < 54) &
        (bg_points[:,2] > -5) &
        (bg_points[:,2] < 3)
    ]

    return bg_points, obj_points


if __name__ == '__main__':

    if len(sys.argv) != 3:
        scene_begin = 0
        scene_end = 850
    else:
        scene_begin = int(sys.argv[1])
        scene_end = int(sys.argv[2])


    dataset_path = '../../data/nuscenes'
    output_path = '../../data/nuscenes_agg'
    os.makedirs(output_path, exist_ok=True)

    nusc = NuScenes(version='v1.0-trainval', dataroot=dataset_path)

    infos = get_nuscenes_infos( nusc )
    with open(f'{output_path}/nuscenes_infos_trainval.pkl', 'wb') as f:
        pickle.dump(infos, f)

    aggregate_nuscenes( nusc, infos, begin=scene_begin, end=scene_end, output_path=output_path )
