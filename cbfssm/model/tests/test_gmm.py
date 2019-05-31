import numpy as np
import tensorflow as tf
from scipy import stats

from cbfssm.model.gmm_utils import *


class GMM(object):
    def __init__(self, weights, locs, variances):
        self.weights = np.atleast_1d(weights)
        self.gaussians = [stats.multivariate_normal(loc, variance)
                          for loc, variance in zip(locs, variances)]

    def __len__(self):
        return len(self.gaussians)

    @property
    def mean(self):
        mean = 0
        for weight, gaussian in zip(self.weights, self.gaussians):
            mean = mean + weight * gaussian.mean
        return mean.squeeze()

    @property
    def variance(self):
        # https://stats.stackexchange.com/questions/16608/what-is-the-variance-of-the-weighted-mixture-of-two-gaussians
        d = self.gaussians[0].dim
        variance = np.zeros((d, d))
        for weight, gaussian in zip(self.weights, self.gaussians):
            variance += weight * gaussian.cov
            variance += weight * np.outer(gaussian.mean, gaussian.mean)

        mean = self.mean
        variance -= np.outer(mean, mean)

        return variance.squeeze()

    @property
    def std(self):
        return np.sqrt(self.variance)

    def pdf(self, x):
        x = np.asarray(x)
        x = np.atleast_1d(x.squeeze())

        prob = 0
        for weight, gaussian in zip(self.weights, self.gaussians):
            prob += weight * gaussian.pdf(x)
        return prob


def numpy_condition_gaussian_on_gmm(gaussian, gmm):
    """Conditions the gaussian distribution on the GMM model.

    Assumes that both distributions are diagonal!

    The result is a GMM again, see
    http://blog.jafma.net/2010/11/09/the-product-of-two-gaussian-pdfs-is-not-a-pdf-but-is-gaussian-a-k-a-loving-algebra/
    """
    if isinstance(gaussian, stats._multivariate.multivariate_normal_frozen):
        gauss_mean = np.atleast_2d(gaussian.mean)
        gauss_var = np.atleast_2d(np.diag(gaussian.cov))
    else:
        gauss_mean = np.atleast_2d(gaussian.mean())
        gauss_var = np.atleast_2d(gaussian.var())

    gmm_means = []
    gmm_vars = []
    for gaussian in gmm.gaussians:
        gmm_means.append(gaussian.mean)
        gmm_vars.append(np.diag(gaussian.cov))

    gmm_means = np.stack(gmm_means)
    gmm_vars = np.stack(gmm_vars)

    var_sum = gmm_vars + gauss_var
    var_product = gmm_vars * gauss_var
    new_vars = var_product / var_sum

    new_means = gmm_vars * gauss_mean + gmm_means * gauss_var
    new_means /= var_sum

    exponent = -np.sum(np.square(gmm_means - gauss_mean) / (2 * var_sum), axis=-1)
    constant = gmm.weights / np.sqrt(2 * np.pi * np.prod(var_sum, axis=-1))

    new_weights = constant * np.exp(exponent)
    new_weights /= np.sum(new_weights)

    return GMM(locs=new_means, variances=new_vars, weights=new_weights)


def test_conditioning_against_numpy():
    gmm_means = np.array([[-1.2, -0.9, 0.3],
                          [1.1, 0.3, 0.2]])
    gmm_variances = np.array([[0.3, 0.2, 0.6],
                              [0.4, 0.1, 0.23]])

    prior_means = np.random.rand(4, 3)
    prior_variances = 0.2 + np.random.rand(4, 3)

    gmm = GMM(weights=[0.5, 0.5], locs=gmm_means, variances=gmm_variances)

    posterior_means = []
    posterior_variances = []
    posterior_weights = []
    for prior_mean, prior_variance in zip(prior_means, prior_variances):
        gaussian = stats.multivariate_normal(prior_mean, prior_variance)
        posterior = numpy_condition_gaussian_on_gmm(gaussian, gmm)

        means = np.stack([g.mean for g in posterior.gaussians])
        variances = np.stack([np.diag(g.cov) for g in posterior.gaussians])

        posterior_means.append(means)
        posterior_variances.append(variances)
        posterior_weights.append(posterior.weights)

    with tf.Graph().as_default():
        tf_means, tf_vars, tf_weights = condition_diag_gaussian_on_diag_gmm(
            prior_means, prior_variances, gmm_means, gmm_variances,
            gmm_weights=gmm.weights
        )
        with tf.Session() as session:
            tf_means, tf_vars, tf_weights = session.run([tf_means, tf_vars, tf_weights])

    np.testing.assert_allclose(tf_means, np.stack(posterior_means))
    np.testing.assert_allclose(tf_vars, np.stack(posterior_variances))
    np.testing.assert_allclose(tf_weights, np.stack(posterior_weights))


def test_conitioning_gaussian():
    gmm_means = np.array([[-1.2, -0.9, 0.3]])
    gmm_variance = np.array([[0.3, 0.2, 0.6]])

    prior_mean = np.random.rand(1, 3)
    prior_variance = 0.2 + np.random.rand(1, 3)

    with tf.Graph().as_default():
        tf_means, tf_vars, tf_weights = condition_diag_gaussian_on_diag_gmm(
            prior_mean, prior_variance, gmm_means, gmm_variance)
        with tf.Session() as session:
            tf_means, tf_vars, tf_weights = session.run([tf_means, tf_vars, tf_weights])

    # ground truth
    gmm_covariance = np.diag(gmm_variance[0])
    prior_covariance = np.diag(prior_variance[0])

    true_post_covariance = np.linalg.inv(
        np.linalg.inv(prior_covariance) + np.linalg.inv(gmm_covariance))

    m1 = np.linalg.inv(prior_covariance).dot(prior_mean.T)
    m2 = np.linalg.inv(gmm_covariance).dot(gmm_means.T)

    np.testing.assert_allclose(tf_vars[0, 0], np.diag(true_post_covariance))
    np.testing.assert_allclose(tf_means[0], true_post_covariance.dot(m1 + m2).T)


def test_moments_vs_numpy():
    gmm_means = np.array([[-1.2, -0.9, 0.3],
                          [1.1, 0.3, 0.2]])
    gmm_variances = np.array([[0.3, 0.2, 0.6],
                              [0.4, 0.1, 0.23]])

    gmm = GMM(weights=[0.2, 0.8], locs=gmm_means, variances=gmm_variances)

    with tf.Graph().as_default():
        mean, var = compute_gmm_moments(gmm_means, gmm_variances, gmm.weights)
        with tf.Session() as session:
            mean, var = session.run([mean, var])

    np.testing.assert_allclose(mean, gmm.mean)
    np.testing.assert_allclose(var, gmm.variance)


def test_moments_single_gaussian():
    gmm_means = np.array([[-1.2, -0.9, 0.3]])
    gmm_variances = np.array([[0.3, 0.2, 0.6]])

    gmm = GMM(weights=[0.2, 0.8], locs=gmm_means, variances=gmm_variances)

    with tf.Graph().as_default():
        mean, var = compute_gmm_moments(gmm_means, gmm_variances, gmm.weights)
        with tf.Session() as session:
            mean, var = session.run([mean, var])

    np.testing.assert_allclose(mean, gmm_means[0])
    np.testing.assert_allclose(var, np.diag(gmm_variances[0]), rtol=0, atol=1e-10)


def test_diag_moments():
    gmm_means = np.array([[-1.2, -0.9, 0.3],
                          [1.1, 0.3, 0.2]])
    gmm_variances = np.array([[0.3, 0.2, 0.6],
                              [0.4, 0.1, 0.23]])

    gmm = GMM(weights=[0.2, 0.8], locs=gmm_means, variances=gmm_variances)

    with tf.Graph().as_default():
        mean, var = compute_gmm_moments(gmm_means, gmm_variances, gmm.weights)
        var_diag_part = batch_diag_part(var)
        diag_mean, diag_var = compute_gmm_moments_diag(gmm_means, gmm_variances, gmm.weights)
        with tf.Session() as session:
            mean, var, diag_mean, diag_var = session.run([mean, var_diag_part, diag_mean, diag_var])

    np.testing.assert_allclose(mean, diag_mean)
    np.testing.assert_allclose(var, diag_var)


if __name__ == '__main__':
    test_conditioning_against_numpy()
    test_conitioning_gaussian()
    test_moments_vs_numpy()
    test_moments_single_gaussian()
    test_diag_moments()
