import math

import numpy as np


DATA_MIN = -140
DATA_MAX = 260


def cut(data, mid, size):
    mid_x, mid_y, mid_z = mid
    min_x = max(0, math.floor(mid_x - size // 2))
    max_x = min(math.ceil(mid_x + size // 2), data.shape[0]-1)
    min_y = max(0, math.floor(mid_y - size // 2))
    max_y = min(math.ceil(mid_y + size // 2), data.shape[1]-1)
    min_z = max(0, math.floor(mid_z - size // 2))
    max_z = min(math.ceil(mid_z + size // 2), data.shape[2]-1)
    data = data[min_x:max_x, min_y:max_y, min_z:max_z]
    return data


def normalize(data):
    data[data > DATA_MAX] = DATA_MAX
    data[data < DATA_MIN] = DATA_MIN
    data = (data - DATA_MIN) / (DATA_MAX - DATA_MIN)
    return data


def find_mid(data):
    non_zero = np.nonzero(data)
    x = non_zero[0]
    y = non_zero[1]
    z = non_zero[2]
    mid_z = (max(z) + min(z)) // 2
    mid_y = (max(y) + min(y)) // 2
    mid_x = (max(x) + min(x)) // 2
    return (mid_x, mid_y, mid_z)


def padding(data, size):
    x, y, z = data.shape
    img = np.zeros((size, size, size))
    x_diff = (size-x) // 2
    y_diff = (size-y) // 2
    z_diff = (size-z) // 2
    img[x_diff:x_diff+x, y_diff:y_diff+y, z_diff:z_diff+z] = data
    return img


def broadcast_to_shape(data, shape):
    x = data.shape[0] == shape
    y = data.shape[1] == shape
    z = data.shape[2] == shape
    if not x:
        diff = (shape - data.shape[0])
        data = np.pad(data, [(diff // 2, diff - (diff // 2)),(0,0),(0,0)], mode = "constant")
    if not y:
        diff = (shape - data.shape[1])
        data = np.pad(data, [(0,0),(diff // 2, diff - (diff // 2)),(0,0)], mode = "constant")
    if not x:
        diff = (shape - data.shape[2])
        data = np.pad(data, [(0,0),(0,0),(diff // 2, diff - (diff // 2))], mode = "constant")
    return data
