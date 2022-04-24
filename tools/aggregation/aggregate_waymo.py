import os
import sys
import pickle
import numpy as np
import tensorflow as tf

from glob import glob
from tqdm.autonotebook import tqdm

from waymo_open_dataset import dataset_pb2
from mmcv.fileio import FileClient
from mmdet3d.core.bbox import (get_box_type, CameraInstance3DBoxes)
from mmdet3d.core.bbox.box_np_ops import points_in_rbbox
from mmdet3d.datasets.pipelines.loading import LoadPointsFromFile
from utils import rot_matrix_2d, extract_object_point_clouds

file_client = FileClient(backend='disk')
_load_points_from_file = LoadPointsFromFile(
    coord_type='LIDAR',
    load_dim=6,
    use_dim=6,
    file_client_args=dict(backend='disk')
)

selected_waymo_classes = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']
type_list = [
    'UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST'
]


def load_points_from_file(filename):
    results = dict(pts_filename=filename)
    _load_points_from_file(results)
    return results['points'].tensor.numpy()


def convert_kitti_boxes(annos, calib):
    rect = calib['R0_rect'].astype(np.float32)
    Trv2c = calib['Tr_velo_to_cam'].astype(np.float32)
    box_type_3d, box_mode_3d = get_box_type('LiDAR')
    gt_boxes = CameraInstance3DBoxes(
        np.concatenate([
            annos['location'], annos['dimensions'], annos['rotation_y'][...,np.newaxis]
        ], axis=1).astype(np.float32)
    ).convert_to(box_mode_3d, np.linalg.inv(rect @ Trv2c))
    return gt_boxes.tensor.numpy()


def get_waymo_infos( dataset_path ):
    with open(f'{dataset_path}/kitti_format/waymo_infos_trainval.pkl', 'rb') as f:
        infos = {info['image']['image_idx']: info for info in pickle.load(f)}
    
    print('Loading waymo meta infos:')
    for prefix, split in enumerate(['training', 'validation']):
        tfrecords = sorted(glob(f'{dataset_path}/{split}/*.tfrecord'))
        for seq_id, tfrecord in enumerate(tqdm(tfrecords, desc=f'{split} split')):
            seq_id = int(seq_id + prefix * 1e3)
            num_frames = len(list(filter(lambda idx: idx//1e3 == seq_id, infos.keys())))
            dataset = tf.data.TFRecordDataset(tfrecord, compression_type='')
            for frame_id, data in enumerate(tqdm(dataset, total=num_frames, desc=f'Seq {seq_id}', leave=False)):
                frame_id = int(seq_id*1e3 + frame_id)
                frame = dataset_pb2.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                info = infos[frame_id]
                info['annos']['obj_ids'] = []
                for label in frame.laser_labels:
                    if type_list[label.type] not in selected_waymo_classes:
                        continue
                    if label.num_lidar_points_in_box < 1:
                        continue
                    info['annos']['obj_ids'].append(label.id)

    return infos


def aggregate_waymo_sequence( dataset, seq_id, output_path=None ):
    dataset_path = dataset['dataset_path']
    split = dataset['split']

    if 'infos' not in dataset:
        dataset['infos'] = get_waymo_infos(dataset_path)
    infos = dataset['infos']

    if 'tfrecords' not in dataset:
        dataset['tfrecords'] = sorted(glob(f'{dataset_path}/{split}/*.tfrecord'))
    sequence = tf.data.TFRecordDataset(dataset['tfrecords'][seq_id], compression_type='')
    seq_id = int(seq_id + {'training': 0, 'validation': 1}[split] * 1e3)

    obj_points = {}
    bg_points = []

    num_frames = len(list(filter(lambda idx: idx//1e3 == seq_id, infos.keys())))
    for frame_id, data in enumerate(tqdm(sequence, total=num_frames, desc=f'Seq {seq_id}', leave=False)):
        frame_id = int(seq_id*1e3 + frame_id)
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        info = infos[frame_id]
        annos = info['annos']

        ######################################################################## 
        # GT boxes for current frame
        ######################################################################## 
        # if 'obj_ids' not in annos:
        annos['obj_ids'] = []
        for label in frame.laser_labels:
            if type_list[label.type] not in selected_waymo_classes:
                continue
            if label.num_lidar_points_in_box < 1:
                continue
            annos['obj_ids'].append(label.id)
        gt_boxes = convert_kitti_boxes(annos, info['calib'])
        boxes = { obj_id: box for obj_id, box in zip(annos['obj_ids'], gt_boxes) }

        ######################################################################## 
        # Load frame point cloud
        ######################################################################## 
        points = load_points_from_file(f'{dataset_path}/kitti_format/training/velodyne/{frame_id:0>7}.bin')
        points = np.concatenate([points, np.full((points.shape[0], 1), frame_id)], axis=1)

        # Remove noisy points near the LiDAR sensor
        # points = points[np.linalg.norm(points[:,:3], axis=-1) > 2]

        ######################################################################## 
        # Extract and aggregate object point clouds
        ######################################################################## 
        obj_mask, obj_points = extract_object_point_clouds(points, boxes, origin=(0.5, 0.5, 0.), obj_points=obj_points)

        ######################################################################## 
        # Transform and aggregate background point clouds
        ######################################################################## 
        points = points[obj_mask.sum(-1) == 0]
        # if 'pose' not in calib:
        #     calib['pose'] = np.array([x for x in frame.pose.transform]).reshape((4,4))
        points[:,:3] = np.dot(
            np.concatenate([points[:,:3], np.ones((points.shape[0], 1))], axis=1),
            info['pose'].T
        )[:,:3]
        
        bg_points.append(points)

    bg_points = np.concatenate(bg_points)
    
    ############################################################################
    # Save scene and object point clouds
    ############################################################################
    if output_path is not None:
        os.makedirs(f'{output_path}/sequences', exist_ok=True)
        os.makedirs(f'{output_path}/objects', exist_ok=True)
        file_client.put(bg_points.flatten().tobytes(), f'{output_path}/sequences/{seq_id}.bin')
        for obj_id, obj_points_i in obj_points.items():
            file_client.put(obj_points_i.flatten().tobytes(), f'{output_path}/objects/{obj_id}.bin')
    else:
        return bg_points, obj_points


def aggregate_waymo( dataset, begin=0, end=798, output_path=None ):
    dataset_path = dataset['dataset_path']
    split = dataset['split']
    infos = dataset['infos']
    dataset['tfrecords'] = sorted(glob(f'{dataset_path}/{split}/*.tfrecord'))

    for seq_id in tqdm(range(begin, end), desc=f'{split} split'):
        aggregate_waymo_sequence( dataset, seq_id, output_path )


def load_waymo_points_with_object_mask( dataset_path, info ):
    frame_id = info['image']['image_idx']
    points = load_points_from_file(f'{dataset_path}/kitti_format/training/velodyne/{frame_id:0>7}.bin')
    gt_boxes = convert_kitti_boxes(info['annos'], info['calib'])
    boxes = { obj_id: box for obj_id, box in zip(info['annos']['obj_ids'], gt_boxes) }
    obj_mask = points_in_rbbox(points, np.stack(list(boxes.values())), origin=(0.5, 0.5, 0.))
    return points, obj_mask


def load_waymo_aggregated_points( dataset_path, info ):
    return np.zeros((0,6), dtype=np.float32), {}


if __name__ == '__main__':
    
    split = sys.argv[1]
    assert(split in ['training', 'validation'])

    if len(sys.argv) != 4:
        seq_begin = 0
        seq_end = 798 if split == 'training' else 202
    else:
        seq_begin = int(sys.argv[2])
        seq_end = int(sys.argv[3])


    dataset_path = '../../data/waymo'
    output_path = '../../data/waymo_agg'
    os.makedirs(output_path, exist_ok=True)

    infos_file = f'{output_path}/waymo_infos_trainval.pkl'
    if os.path.isfile(infos_file):
        with open(infos_file, 'rb') as f:
            infos = pickle.load(f)
    else:
        infos = get_waymo_infos( dataset_path )
        with open(infos_file, 'wb') as f:
            pickle.dump(infos, f)
    
    dataset = dict(
        dataset_path=dataset_path,
        split=split,
        infos=infos
    )

    aggregate_waymo( dataset, begin=seq_begin, end=seq_end, output_path=output_path )
