import numpy as np
import PIL
import torch
import torch.nn.functional as F
from pyquaternion import Quaternion


def convert_egopose_to_matrix_numpy(trans, rot):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = Quaternion(rot).rotation_matrix
    translation = np.array(trans)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix


def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix