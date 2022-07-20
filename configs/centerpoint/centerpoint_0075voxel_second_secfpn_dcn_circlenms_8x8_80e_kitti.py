_base_ = [
    '../_base_/datasets/kitti-3d-cbgs.py',
    '../_base_/models/centerpoint_0075voxel_second_secfpn_dcn_circlenms_kitti.py',
    '../_base_/schedules/cyclic_80e.py',
    '../_base_/default_runtime.py'
]

# point_cloud_range = [0, -39.75, -3, 69.75, 39.75, 1]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]

model = dict(
    pts_voxel_layer=dict(point_cloud_range=point_cloud_range),
    pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2])),
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range)),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2])))
