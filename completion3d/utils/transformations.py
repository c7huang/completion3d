import numpy as np
try:
    from numpy.typing import ArrayLike
except:
    from typing import Union
    ArrayLike = Union[tuple, list, np.ndarray]
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interpn

def cartesian_to_homogeneous(x: ArrayLike) -> np.ndarray:
    """Convert K-dimensional points/vectors to homoveneous coordinates

    :param x: a (N, K) array representing N K-dimensional points/vectors
    :type x: array_like
    :returns: a (N, K+1) array in homogeneous coordinates
    :rtype: np.ndarray
    """
    x = np.asarray(x)
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


def rotate2d(x: ArrayLike, angle: ArrayLike) -> np.ndarray:
    """Rotate an array of 2D points/vectors counterclockwise

    :param x: an (N, 2) array representing N 2D points/vectors
    :type x: array_like
    :param angle: the angle to be rotated in radian
    :type angle: float
    :returns: an (N, 2) np.ndarray with the points/vectors rotated
    :rtype: np.ndarray
    """

    sin, cos = np.sin(angle), np.cos(angle)
    rot = np.array([[cos, -sin], [sin, cos]])
    return np.dot(np.asarray(x)[:,:2], rot.T)


def transformation3d_from_rotation_translation(
    rotation_matrix: ArrayLike = np.identity(3),
    translation: ArrayLike = np.zeros(3)
) -> np.ndarray: 
    """Compute a 3d transformation matrix (4x4) from a rotation matrix and
    translation vector.

    :param rotation_matrix: a 3x3 rotation matrix
    :type rotation_matrix: array_like
    :param translation: translation vector
    :type translation: array_like
    :returns: a 4x4 transformation matrix
    :rtype: np.ndarray
    """
    transformation = np.identity(4)
    transformation[:3,:3] = np.asarray(rotation_matrix)
    transformation[:3,3] = np.asarray(translation)
    return transformation


def transformation3d_from_quaternion_translation(
    quaternion: ArrayLike, translation: ArrayLike = np.zeros(3)
) -> np.ndarray:
    """Compute a 3D transformation matrix (4x4) from a quaternion and
    translation vector.

    :param quaternion: a quaternion in (x, y, z, w) format
    :type quaternion: array_like
    :param translation: translation vector
    :type translation: array_like
    :returns: a 4x4 transformation matrix
    :rtype: np.ndarray
    """
    return transformation3d_from_rotation_translation(
        R.from_quat(quaternion).as_matrix(), translation
    )


def transform3d(x: ArrayLike, transformation: ArrayLike = np.identity(4)) -> np.ndarray:
    """Apply 3D transformation to an array of 3D points/vectors
    
    :param x: an (N, 3) array representing N 3D points/vectors
    :type x: array_like
    :param transformation: a 4x4 transformation matrix
    :type transformation: array_like
    :returns: an (N, 3) array with the points/vectors transformed
    :rtype: np.ndarray
    """
    result = np.dot(cartesian_to_homogeneous(np.asarray(x)[:,:3]), transformation.T)
    return result[:,:3] / result[:,3:4]


def interpolate(image: ArrayLike, points: ArrayLike, method: str ='linear') -> np.ndarray:
    return interpn( 
        (np.arange(image.shape[0]), np.arange(image.shape[1])),
        image,
        (points[:,0], points[:,1]),
        method=method
    )