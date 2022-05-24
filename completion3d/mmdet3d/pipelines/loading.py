import pickle
import numpy as np

from mmdet.datasets.builder import PIPELINES
from ...aggregation.common import load_aggregated_points
from ...utils.transformations import transformation3d_with_translation


@PIPELINES.register_module()
class LoadAggregatedPoints(object):
    def __init__(self, agg_dataset_path, dbinfos_path, agginfos_path,
        num_point_features, use_point_features=None, point_cloud_range=None
    ):
        self.agg_dataset_path = agg_dataset_path

        with open(dbinfos_path, 'rb') as f:
            dbinfos = pickle.load(f)

        with open(agginfos_path, 'rb') as f:
            self.agginfos = pickle.load(f)

        self.num_point_features = num_point_features
        self.use_point_features = use_point_features
        self.point_cloud_range = point_cloud_range

        self.group_id2object_id = {}
        for infos in dbinfos.values():
            for info in infos:
                if info['group_id'] in self.group_id2object_id:
                    raise
                self.group_id2object_id[info['group_id']] = \
                    self.agginfos[info['image_idx']]['object_ids'][info['gt_idx']]


    def __call__(self, input_dict):
        agginfo = self.agginfos[input_dict['sample_idx']]
        scene_id = agginfo['scene_id']
        scene_transformation = agginfo['scene_transformation']

        # Apply transformations from augmentation
        if 'transformation_3d_flow' in input_dict:
            for t in input_dict['transformation_3d_flow']:
                if t == 'R':
                    scene_transformation = transformation3d_with_translation(
                        transformation=np.linalg.inv(input_dict['pcd_rotation'])
                    ) @ scene_transformation
                elif t == 'S':
                    scene_transformation = transformation3d_with_translation(
                        transformation=np.identity(3)*input_dict['pcd_scale_factor']
                    ) @ scene_transformation
                elif t == 'T':
                    scene_transformation = transformation3d_with_translation(
                        translation=input_dict['pcd_trans']
                    ) @ scene_transformation
                elif t == 'HF':
                    horizontal_flip = np.identity(4)
                    horizontal_flip[1,1] = -1
                    scene_transformation = horizontal_flip @ scene_transformation
                elif t == 'VF':
                    vertical_flip = np.identity(4)
                    vertical_flip[0,0] = -1
                    scene_transformation = vertical_flip @ scene_transformation
        
        # Get objects from dbsample augmentation
        object_ids = agginfo['object_ids']
        if 'dbsample_group_ids' in input_dict:
            object_ids += [self.group_id2object_id[group_id] for group_id in input_dict['dbsample_group_ids']]

        # print(scene_id)
        points_agg = load_aggregated_points(
            agg_dataset_path = self.agg_dataset_path,
            scene_id = scene_id,
            object_ids = object_ids,
            gt_boxes = input_dict['gt_bboxes_3d'].tensor.numpy()[:,:7],
            scene_transformation = scene_transformation,
            num_point_features = self.num_point_features,
            use_point_features = self.use_point_features,
            point_cloud_range = self.point_cloud_range,
            combine = True
        )

        input_dict['points_agg'] = points_agg

        # DEBUG: plot and compare the point clouds after augmentation
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(24,12))
        # plt.subplot(1,2,1)
        # plt.scatter(input_dict['points'][:,0], input_dict['points'][:,1], s=100/input_dict['points'].shape[0])
        # plt.xlim((self.point_cloud_range[0], self.point_cloud_range[3]))
        # plt.ylim((self.point_cloud_range[1], self.point_cloud_range[4]))

        # plt.subplot(1,2,2)
        # plt.scatter(points_agg[:,0], points_agg[:,1], s=100/points_agg.shape[0])
        # plt.xlim((self.point_cloud_range[0], self.point_cloud_range[3]))
        # plt.ylim((self.point_cloud_range[1], self.point_cloud_range[4]))

        # plt.tight_layout()
        # plt.savefig(f'{input_dict["sample_idx"]}.png')
        # plt.close()
