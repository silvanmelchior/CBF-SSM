"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
@author: Andreas Doerr

The following snippets are derived from GPFlow V 1.0
  (https://github.com/GPflow/GPflow)
Copyright 2017 st--, Mark van der Wilk, licensed under the Apache License, Version 2.0,
cf. 3rd-party-licenses.txt file in the root directory of this source tree.
"""

import tensorflow as tf
from cbfssm.model.tf_transform import forward, backward


class RBF:

    def __init__(self, variance, lengthscales, dtype=tf.float32):

        with tf.name_scope('kern'):
            self.variance_unc = tf.Variable(backward(variance),
                                            dtype=dtype)
            self.variance = forward(self.variance_unc)

            self.lengthscales_unc = tf.Variable(backward(lengthscales),
                                                dtype=dtype)
            self.lengthscales = forward(self.lengthscales_unc)

    def square_dist(self, X, X2):
        X = X / self.lengthscales
        Xs = tf.reduce_sum(tf.square(X), 1)
        if X2 is None:
            return -2 * tf.matmul(X, X, transpose_b=True) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
        else:
            X2 = X2 / self.lengthscales
            X2s = tf.reduce_sum(tf.square(X2), 1)
            return -2 * tf.matmul(X, X2, transpose_b=True) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))

    def Kdiag(self, X):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))

    def K(self, X, X2=None):
        return self.variance * tf.exp(-self.square_dist(X, X2) / 2)


def cast_cholesky(mat, jitter=1e-8):
    dtype = mat.dtype
    jitter = tf.eye(mat.shape.as_list()[0], dtype=tf.float64) * jitter
    if dtype == tf.float64:
        return tf.cholesky(mat + jitter)
    else:
        mat = tf.cast(mat, tf.float64)
        chol = tf.cholesky(mat + jitter)
        return tf.cast(chol, dtype)


def conditional(Xnew, X, kern, f, q_sqrt, Lm=None):
    # compute kernel stuff
    num_data = tf.shape(X)[0]
    num_func = tf.shape(f)[1]
    Kmn = kern.K(X, Xnew)
    if Lm is None:
        Kmm = kern.K(X) + tf.eye(num_data, X.dtype) * 1e-8
        Lm = cast_cholesky(Kmm)


    # Compute the projection matrix A
    A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)

    # compute the covariance due to the conditioning
    fvar = kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
    shape = tf.stack([num_func, 1])
    fvar = tf.tile(tf.expand_dims(fvar, 0), shape)

    # another backsubstitution in the unwhitened case
    A = tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False)

    # construct the conditional mean
    fmean = tf.matmul(A, f, transpose_a=True)

    if q_sqrt is not None:
        if q_sqrt.get_shape().ndims == 2:
            LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)
        elif q_sqrt.get_shape().ndims == 3:
            L = q_sqrt
            A_tiled = tf.tile(tf.expand_dims(A, 0), tf.stack([num_func, 1, 1]))
            LTA = tf.matmul(L, A_tiled, transpose_a=True)
        else:
            raise ValueError("bad dimension for q_sqrt")

        fvar = fvar + tf.reduce_sum(tf.square(LTA), 1)

    fvar = tf.transpose(fvar)

    return fmean, fvar
