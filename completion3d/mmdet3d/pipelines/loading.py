import pickle
import numpy as np
import torch

from typing import Optional, Union
try:
    from numpy.typing import ArrayLike
except:
    ArrayLike = Union[tuple, list, np.ndarray]
from time import perf_counter
from mmcv.utils import get_logger
from mmdet.datasets.builder import PIPELINES
from mmdet3d.core import LiDARInstance3DBoxes, LiDARPoints
from mmdet3d.datasets.pipelines import LoadPointsFromFile
from ...aggregation.common import load_aggregated_points
from ...utils.transforms import affine_transform


@PIPELINES.register_module()
class UnloadAnnotations3D(object):
    ann_fields = [
        'img_info', 'ann_info',
        'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
        'gt_masks', 'gt_semantic_seg',
        'gt_bboxes_3d', 'centers2d', 'depths',
        'gt_labels_3d', 'attr_labels',
        'pts_instance_mask', 'pts_semantic_mask'
    ]

    meta_fields = [
        'bbox_fields', 'mask_fields', 'seg_fields',
        'bbox3d_fields', 'pts_mask_fields', 'pts_seg_fields'
    ]

    def __call__(self, results):
        for key in self.ann_fields:
            if key in results:
                del results[key]
        for key in self.meta_fields:
            results[key] = []
        return results


@PIPELINES.register_module()
class RemapLabels(object):
    def __init__(self, label_map, label_key='gt_labels_3d'):
        self.label_key = label_key
        self.label_map = label_map

    def __call__(self, results):
        results[self.label_key] = np.array([
            self.label_map[label] for label in results[self.label_key]
        ], dtype=results[self.label_key].dtype)
        return results


@PIPELINES.register_module()
class LoadDummyPoints(LoadPointsFromFile):
    def _load_points(self, pts_filename):
        return np.zeros((0, self.load_dim), dtype=np.float32)


@PIPELINES.register_module()
class LoadAggregatedPoints(object):
    """Load the aggregated version of the point cloud for the current frame.

    This pipeline expects the following fields to be loaded before it:
        * gt_bboxes_3d           (from LoadAnnotations3D)

    If any augmentation is applied (e.g., ObjectSample, GlobalRotScaleTrans,
    RandomFlip3D), they should be applied before this pipeline as well.

    :param scene_path: the location containing the aggregated scene point
        clouds. The point clouds are stored in bytes and can be loaded into
        float32 numpy arrays.
    :type scene_path: str
    :param object_path: the location containing the aggregated object point
        clouds. The point clouds are stored in bytes and can be loaded into
        float32 numpy arrays.
    :type object_path: str
    :param agginfos_path: the agginfos file generated using
        `completion3d.aggregation.xxx_aggregation`
    :type agginfos_path: str
    :param dbinfos_path: the dbinfos file generated during the mmdetection3d
        data preparation process
    :type dbinfos_path: str
    :param num_point_features: the number of features in the aggregated point
        clouds. This should be 14 for nuscenes and 15 for Waymo
    :type num_point_features: int
    :param use_point_features: the indices of the point features that will be
        kept and used after loading
    :type use_point_features: ArrayLike | int | None
    :param box_origin: the relative location of the bounding box center inside
        the object
    :type box_origin: ArrayLike | None
    :param point_cloud_range: the range of the point cloud in 
        `[x_min, y_min, z_min, x_max, y_max, z_max]` format.
        The points outside of this range will be discarded.
    :type point_cloud_range: ArrayLike | None
    :param compression: if the aggregated dataset is in compressed format (e.g.,
        zlib), specify the compression algorithm here
    :type compression: bool | str | None
    :param load_as: the key inside the dictionary the aggregated point cloud
        will be stored in. Set it to 'points' to replace the original sparse
        point cloud.
    :type load_as: str
    :param load_format: the format the aggregated point cloud will be stored in
        the dictionary. Possible values are: numpy, tensor, mmdet3d
    :type load_format: str
    :param verbose: print additional information
    """

    def __init__(
        self,
        scene_path: str,  object_path: str,
        agginfos_path: str, dbinfos_path: str,
        num_point_features: int,
        use_point_features: Union[ArrayLike, int, None] = None,
        box_origin: Optional[ArrayLike] = (0.5, 0.5, 0.0),
        point_cloud_range: Optional[ArrayLike] = None,
        compression: Union[bool, str, None] = None,
        load_as: str = 'points_agg', load_format: str = 'mmdet3d',
        verbose: bool = False
    ):
        if box_origin is None:
            box_origin = torch.as_tensor([0.5, 0.5, 0.0], dtype=torch.float32)
        if isinstance(use_point_features, int):
            use_point_features = list(range(use_point_features))
        elif use_point_features is None or not isinstance(use_point_features, (list, tuple, np.ndarray)):
            use_point_features = list(range(num_point_features))

        self.scene_path = scene_path
        self.object_path = object_path

        with open(dbinfos_path, 'rb') as f:
            dbinfos = pickle.load(f)

        with open(agginfos_path, 'rb') as f:
            self.agginfos = pickle.load(f)

        self.num_point_features = num_point_features
        self.use_point_features = use_point_features
        self.box_origin_shift = torch.as_tensor(box_origin, dtype=torch.float32) - \
            torch.as_tensor([0.5, 0.5, 0.0], dtype=torch.float32)
        self.point_cloud_range = point_cloud_range
        self.compression = compression
        self.load_as = load_as
        self.load_format = load_format
        self.verbose = verbose

        self.group_id2object_id = {}
        for infos in dbinfos.values():
            for info in infos:
                if info['group_id'] in self.group_id2object_id:
                    raise
                self.group_id2object_id[info['group_id']] = \
                    self.agginfos[info['image_idx']]['object_ids'][info['gt_idx']]


    def __call__(self, input_dict):
        total_start = perf_counter()

        agginfo = self.agginfos[input_dict['sample_idx']]
        gt_boxes = input_dict['gt_bboxes_3d']
        gt_boxes = LiDARInstance3DBoxes(
            tensor = gt_boxes.tensor.clone(), 
            box_dim = gt_boxes.box_dim, 
            with_yaw = gt_boxes.with_yaw
        )

        # Shift box origin based on the aggregated point cloud format
        gt_boxes.tensor[:,:3] += self.box_origin_shift * gt_boxes.tensor[:,3:6]

        # Handle transformation augmentations
        # 1. undo all the transformations to the bounding boxes
        # 2. aggregate the final transformation matrix `aug_transformation`
        aug_transformation = None
        transformation_3d_flow = []
        if 'transformation_3d_flow' in input_dict:
            aug_transformation = np.identity(4)
            transformation_3d_flow = input_dict['transformation_3d_flow']
            for t in reversed(transformation_3d_flow):
                if t == 'R':
                    pcd_rotation = input_dict['pcd_rotation'].T
                    pcd_rotation_angle = np.arctan2(pcd_rotation[0,1], pcd_rotation[0,0])
                    # Undo augmentation
                    gt_boxes.rotate(-pcd_rotation_angle)
                    # Record augmentation
                    aug_transformation = aug_transformation @ \
                        affine_transform(
                            transformation=pcd_rotation
                        )
                elif t == 'S':
                    pcd_scale_factor = input_dict['pcd_scale_factor']
                    # Undo augmentation
                    gt_boxes.scale(1/pcd_scale_factor)
                    # Record augmentation
                    aug_transformation = aug_transformation @ \
                        affine_transform(
                            transformation=np.identity(3)*pcd_scale_factor
                        )
                elif t == 'T':
                    pcd_trans = input_dict['pcd_trans']
                    # Undo augmentation
                    gt_boxes.translate(-pcd_trans)
                    # Record augmentation
                    aug_transformation = aug_transformation @ \
                        affine_transform(
                            translation=pcd_trans
                        )
                elif t == 'HF':
                    # Undo augmentation
                    gt_boxes.flip('horizontal')
                    # Record augmentation
                    horizontal_flip = np.identity(4)
                    horizontal_flip[1,1] = -1
                    aug_transformation = aug_transformation @ horizontal_flip
                elif t == 'VF':
                    # Undo augmentation
                    gt_boxes.flip('vertical')
                    # Record augmentation
                    vertical_flip = np.identity(4)
                    vertical_flip[0,0] = -1
                    aug_transformation = aug_transformation @ vertical_flip

        # Handle GT-sampling augmentation
        object_ids = [id for id in agginfo['object_ids']]
        if 'dbsample_group_ids' in input_dict:
            object_ids += [
                self.group_id2object_id[group_id] \
                for group_id in input_dict['dbsample_group_ids']
            ]

        points_agg = load_aggregated_points(
            scene_path = self.scene_path,
            object_path = self.object_path,
            scene_id = agginfo['scene_id'],
            object_ids = object_ids,
            scene_transform = agginfo['scene_transform'],
            object_transforms = gt_boxes.tensor.numpy()[:,[0,1,2,6]],
            num_point_features = self.num_point_features,
            use_point_features = self.use_point_features,
            scene_chunks = agginfo['scene_chunks'],
            global_transform=aug_transformation,
            point_cloud_range = self.point_cloud_range,
            compression = self.compression,
            combine = True
        )

        if self.load_format == 'numpy':
            input_dict[self.load_as] = points_agg
        elif self.load_format == 'tensor':
            input_dict[self.load_as] = torch.as_tensor(
                points_agg,
                dtype=torch.float32,
                device=torch.device('cpu')
            )
        elif self.load_format == 'mmdet3d':
            input_dict[self.load_as] = LiDARPoints(
                tensor = points_agg,
                points_dim = len(self.use_point_features)
            )
        else:
            raise ValueError(f'unknown format {self.load_format}')

        total_time = perf_counter() - total_start
        if self.verbose:
            logger = get_logger('LoadAggregatedPoints')
            logger.info(f'load time: {total_time:.2f}s')


        ########################################################################
        # DEBUG: plot and compare the point clouds after augmentation
        ########################################################################
        # import matplotlib.pyplot as plt
        # from matplotlib.patches import Rectangle
        # from matplotlib.transforms import Affine2D
        # plt.figure(figsize=(16,8))
        # plt.subplot(1,2,1)
        # plt.scatter(
        #     input_dict['points'][:,0], input_dict['points'][:,1],
        #     s=0.001, alpha=0.5
        # )
        # for box in input_dict['gt_bboxes_3d'].tensor.numpy():
        #     plt.gca().add_patch(Rectangle(
        #         box[:2]-box[3:5]/2, box[3], box[4],
        #         lw=1, ec='tab:orange', fc='tab:orange', alpha=0.25,
        #         transform=Affine2D().rotate_around(box[0], box[1], -box[6]) + 
        #             plt.gca().transData
        #     ))
        # plt.xlim((self.point_cloud_range[0], self.point_cloud_range[3]))
        # plt.ylim((self.point_cloud_range[1], self.point_cloud_range[4]))

        # plt.subplot(1,2,2)
        # plt.scatter(points_agg[:,0], points_agg[:,1], s=0.00001, alpha=0.5)
        # for box in input_dict['gt_bboxes_3d'].tensor.numpy():
        #     plt.gca().add_patch(Rectangle(
        #         box[:2]-box[3:5]/2, box[3], box[4],
        #         lw=1, ec='tab:orange', fc='tab:orange', alpha=0.25,
        #         transform=Affine2D().rotate_around(box[0], box[1], -box[6]) + 
        #             plt.gca().transData
        #     ))
        # plt.xlim((self.point_cloud_range[0], self.point_cloud_range[3]))
        # plt.ylim((self.point_cloud_range[1], self.point_cloud_range[4]))

        # plt.suptitle(
        #     f'R:{pcd_rotation_angle:.2f} '
        #     f'S:{pcd_scale_factor:.2f} '
        #     f'HF:{"HF" in transformation_3d_flow} '
        #     f'VF:{"VF" in transformation_3d_flow}'
        # )
        # plt.tight_layout()
        # plt.savefig(f'{input_dict["sample_idx"]}.png')
        # plt.close()
        # raise
        ########################################################################

        return input_dict
