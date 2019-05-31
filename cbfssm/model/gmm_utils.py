import tensorflow as tf
import numpy as np


__all__ = ["condition_diag_gaussian_on_diag_gmm", "compute_gmm_moments",
           "sample_from_multivariate_gaussian", "compute_gmm_moments_diag"]


def condition_diag_gaussian_on_diag_gmm(mean, var, gmm_means, gmm_vars,
                                        gmm_weights=None):
    """Condition a diagonal Gaussian distribution on a diagonal GMM.

    Parameters
    ----------
    mean : tf.Tensor
        A vector where the last two dimensions (N x d) represent the
        d-dimensional means of N Gaussian distributions.
    var : tf.Tensor
        As mean, but with variance.
    gmm_means : tf.Tensor
        A vector where the last two dimensions (M x d) represent the
        N different d-dimensional means of the gmm mixture components.
    gmm_vars : tf.Tensor
        As gmm_means, but with variances.

    Returns
    -------
    mean : tf.Tensor
        A vector where the last three dimensions (N x M x d) represent
        the represent for each prior Gaussian N the M x d means of the
        resulting mixture distribution
    var : tf.Tensor
        As mean, but with variances
    weights : tf.Tensor
        The last (N x M) components represent the weights of the mixtures.
    """
    num_gmm_clusters = gmm_means.shape[-2].value
    batch_size = var.shape[-2].value

    # Tile to turn both into (N x M x d) vectors
    tile_shape = np.ones(len(mean.shape) + 1)
    tile_shape[-2] = num_gmm_clusters
    mean_tile = tf.tile(tf.expand_dims(mean, -2), tile_shape)
    var_tile = tf.tile(tf.expand_dims(var, -2), tile_shape)

    tile_shape = np.ones(len(gmm_means.shape) + 1)
    tile_shape[-3] = batch_size
    gmm_mean_tile = tf.tile(tf.expand_dims(gmm_means, -3), tile_shape)
    gmm_var_tile = tf.tile(tf.expand_dims(gmm_vars, -3), tile_shape)

    # Intermediate results
    var_sum_inv = 1. / (var_tile + gmm_var_tile)
    var_product = var_tile * gmm_var_tile

    new_gmm_vars = var_product * var_sum_inv

    new_gmm_means = var_tile * gmm_mean_tile + gmm_var_tile * mean_tile
    new_gmm_means *= var_sum_inv

    # Compute the weight
    exponent = -0.5 * tf.reduce_sum(tf.square(gmm_mean_tile - mean_tile) * var_sum_inv,
                                    axis=-1)
    # constant = gmm_weights / tf.sqrt((2 * np.pi) * tf.reduce_prod(var_sum, axis=-1))
    constant = tf.sqrt(tf.reduce_prod(var_sum_inv, axis=-1))
    if gmm_weights is not None:
        constant *= gmm_weights
    exponent += tf.log(constant)

    # Normalize before computing tf.exp
    normalizer = tf.reduce_logsumexp(exponent, axis=-1, keepdims=True)
    new_weights = tf.exp(exponent - normalizer)

    return new_gmm_means, new_gmm_vars, new_weights


def compute_gmm_moments(gmm_means, gmm_variances, gmm_weights, jitter=None,
                        parallel_iterations=2):
    """
    Return the mean and covariance of a GMM distribution.

    Parameters
    ----------
    mean : tf.Tensor
        A vector where the last three dimensions (N x M x d) represent
        the represent for each prior Gaussian N the M x d means of the
        resulting mixture distribution
    var : tf.Tensor
        As mean, but with variances
    weights : tf.Tensor
        The last (N x M) components represent the weights of the mixtures.
    jitter : float, optional
        An optional jitter added to the diagonal entry of the covariance.
    parallel_iterations : inf, optional
        How many parallel iterations to conduct in map_fn.

    Returns
    -------
    mean : tf.Tensor
        The last two (N x d) dimension represent the mean of the GMM.
    covariance : tf.Tensor
        The last three (N x d x d) dimensions represent the covariance
        matrix of the GMM.
    """
    weights = tf.expand_dims(gmm_weights, axis=-1)

    # Compute gmm mean
    weighted_means = gmm_means * weights
    mean = tf.reduce_sum(weighted_means, axis=-2)

    # Compute the average cluster-mean covariance
    gmm_means_matrix = tf.expand_dims(gmm_means * tf.sqrt(weights), axis=-1)
    gmm_weighted_mean_covar = tf.matmul(gmm_means_matrix, gmm_means_matrix,
                                        transpose_b=True)
    gmm_means_covar = tf.reduce_sum(gmm_weighted_mean_covar, axis=-3)

    # Compute gmm-mean covariance
    gmm_mean_matrix = tf.expand_dims(mean, axis=-1)
    gmm_mean_covariance = tf.matmul(gmm_mean_matrix, gmm_mean_matrix, transpose_b=True)

    # Compute the average cluster variance
    if jitter is not None:
        gmm_variances += jitter
    average_variance = tf.reduce_sum(gmm_variances * weights, axis=-2)
    # Convert to diagonal matrix
    average_covariance = tf.matrix_diag(average_variance)

    covariance = average_covariance + gmm_means_covar - gmm_mean_covariance

    return mean, covariance


def compute_gmm_moments_diag(gmm_means, gmm_variances, gmm_weights):
    """
    Return the mean and covariance of a GMM distribution.

    Parameters
    ----------
    mean : tf.Tensor
        A vector where the last three dimensions (N x M x d) represent
        the represent for each prior Gaussian N the M x d means of the
        resulting mixture distribution
    var : tf.Tensor
        As mean, but with variances
    weights : tf.Tensor
        The last (N x M) components represent the weights of the mixtures.

    Returns
    -------
    mean : tf.Tensor
        The last two (N x d) dimension represent the mean of the GMM.
    variance : tf.Tensor
        The last tow (N x d) dimensions represent the variance
        matrix of the GMM.
    """
    weights = tf.expand_dims(gmm_weights, axis=-1)

    # Compute gmm mean
    weighted_means = gmm_means * weights
    mean = tf.reduce_sum(weighted_means, axis=-2)

    # Compute the average cluster-mean covariance
    gmm_means_covar = tf.square(gmm_means) * weights
    gmm_means_covar = tf.reduce_sum(gmm_means_covar, axis=-2)

    # Compute gmm-mean covariance
    gmm_mean_covariance = tf.square(mean)

    # Compute the average cluster variance
    average_variance = tf.reduce_sum(gmm_variances * weights, axis=-2)

    variance = average_variance + gmm_means_covar - gmm_mean_covariance
    return mean, variance


def sample_from_multivariate_gaussian(mean, covariance, eps=None, float64=False):
    """Sample from a multivariate Gaussian distribution.

    The last two dimensions represent the mean, covariance.

    Parameters
    ----------
    mean : tf.Tensor
        The last two (N x d) dimension represent the mean of the GMM.
    covariance : tf.Tensor
        The last three (N x d x d) dimensions represent the covariance
        matrix of the GMM.

    Returns
    -------
    samples : tf.Tensor
        The last (N x d) dimensions represent a sample from the corresponding Gaussian.
    """
    if float64:
        orig_dtype = covariance.dtype
        covariance = tf.cast(covariance, tf.float64)
        cholesky = tf.cholesky(covariance)
        cholesky = tf.cast(cholesky, orig_dtype)
    else:
        cholesky = tf.cholesky(covariance)

    if eps is None:
        eps = tf.random.normal(cholesky.shape[:-1].as_list() + [1, ],
                               dtype=cholesky.dtype)
    delta = tf.matmul(cholesky, eps)
    return mean + tf.squeeze(delta, -1)
