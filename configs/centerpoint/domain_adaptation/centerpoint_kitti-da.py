_base_ = [
    '../../_base_/datasets/kitti-3d-da.py',
    '../../_base_/models/centerpoint_0075voxel_second_secfpn_dcn_circlenms_da.py',
    '../../_base_/schedules/cyclic_80e.py',
    '../../_base_/default_runtime.py'
]

point_cloud_range = [-75, -75, -5, 75, 75, 4]
model = dict(
    pts_voxel_layer=dict(point_cloud_range=point_cloud_range),
    pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2])),
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range)),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2])))
