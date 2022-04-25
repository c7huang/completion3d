import numpy as np
import open3d as o3d
from mmdet3d.core.bbox.box_np_ops import points_in_rbbox

def rot_matrix_2d( angle ):
    sin, cos = np.sin(angle), np.cos(angle)
    return np.array([[cos, -sin], [sin, cos]])


def to_o3d_point_cloud(points):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[:,:3]))


def icp_registration(source, target, max_correspondence_distance):
    source_pc = to_o3d_point_cloud(source)
    target_pc = to_o3d_point_cloud(target)
    reg = o3d.pipelines.registration.registration_icp(
        source_pc, target_pc, max_correspondence_distance
    )
    return reg.transformation


def remove_radius_outlier(points, nb_points, radius, print_progress=False):
    pc = to_o3d_point_cloud(points)
    _, ind = pc.remove_radius_outlier(nb_points, radius, print_progress)
    return points[ind]


def extract_object_point_clouds(
    points, boxes, origin=(0.5, 0.5, 0.),
    obj_points=None, align=False
):
    if obj_points is None:
        obj_points = {}

    if len(boxes) == 0:
        return np.zeros((points.shape[0], 1), dtype=bool), obj_points

    # Identify points in boxes
    # obj_mask: (# points, # boxes)
    obj_mask = points_in_rbbox(points, np.stack(list(boxes.values())), origin=origin)

    for i, (id, box) in enumerate(boxes.items()):

        # Transform object point cloud to object-centered coordinates
        obj_points_i = points[obj_mask[:,i]]
        obj_points_i[:,:3] -= box[:3]
        obj_points_i[:,:2] = np.dot(obj_points_i[:,:2], rot_matrix_2d(box[6]).T)

        # Introduce left-right symmetry to increase point density
        obj_points_i_sym = obj_points_i.copy()
        obj_points_i_sym[:,0] = -obj_points_i_sym[:,0]
        obj_points_i = np.concatenate([obj_points_i, obj_points_i_sym])
        
        # Aggregate object point cloud
        if id in obj_points:
            if not isinstance(obj_points[id], list):
                obj_points[id] = [obj_points[id]]
            if align and obj_points_i.shape[0] > 0 and obj_points[id][-1].shape[0] > 0:
                obj_points_i[:,:3] = np.dot(
                    np.concatenate([obj_points_i[:,:3], np.ones((obj_points_i.shape[0], 1))], axis=1),
                    icp_registration(obj_points_i, obj_points[id][-1], 0.1).T
                )[:,:3]
            obj_points[id].append(obj_points_i)
        else:
            obj_points[id] = obj_points_i

    return obj_mask, obj_points


