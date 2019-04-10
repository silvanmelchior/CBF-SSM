import tensorflow as tf
import numpy as np


class Quaternion:

    @staticmethod
    def multiply(a, b):
        el_0 = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1] - a[..., 2] * b[..., 2] - a[..., 3] * b[..., 3]
        el_1 = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0] + a[..., 2] * b[..., 3] - a[..., 3] * b[..., 2]
        el_2 = a[..., 0] * b[..., 2] - a[..., 1] * b[..., 3] + a[..., 2] * b[..., 0] + a[..., 3] * b[..., 1]
        el_3 = a[..., 0] * b[..., 3] + a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1] + a[..., 3] * b[..., 0]
        return tf.stack((el_0, el_1, el_2, el_3), axis=-1)

    @staticmethod
    def multiply_np(a, b):
        el_0 = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1] - a[..., 2] * b[..., 2] - a[..., 3] * b[..., 3]
        el_1 = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0] + a[..., 2] * b[..., 3] - a[..., 3] * b[..., 2]
        el_2 = a[..., 0] * b[..., 2] - a[..., 1] * b[..., 3] + a[..., 2] * b[..., 0] + a[..., 3] * b[..., 1]
        el_3 = a[..., 0] * b[..., 3] + a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1] + a[..., 3] * b[..., 0]
        return np.stack((el_0, el_1, el_2, el_3), axis=-1)

    @staticmethod
    def invert(a):
        return a * tf.constant([1., -1., -1., -1.], dtype=tf.float64)

    @staticmethod
    def invert_np(a):
        return a * [1., -1., -1., -1.]

    @staticmethod
    def pad_to_quat(a):
        zeros = tf.zeros_like(a, dtype=tf.float64)
        return tf.concat((zeros[..., 0:1], a), axis=-1)

    @staticmethod
    def rot_vec(v, q):
        res = Quaternion.multiply(q, Quaternion.pad_to_quat(v))
        res = Quaternion.multiply(res, Quaternion.invert(q))[..., 1:]
        return res
