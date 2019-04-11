import numpy as np
import tensorflow as tf
from crssm.model.tf_transform import backward, forward
from crssm.model.gp_tf import RBF, conditional
from crssm.model.base_model import BaseModel


class PRSSM(BaseModel):

    def __init__(self, config):
        super(PRSSM, self).__init__(config)

    def _build_graph(self):
        dim_u = self.config['ds'].dim_u
        dim_x = self.config['dim_x']
        dim_y = self.config['ds'].dim_y
        ind_pnt_num = self.config['ind_pnt_num']
        samples = self.config['samples']
        loss_factors = self.config['loss_factors']

        with self.graph.as_default():

            # Variables
            self.zeta_pos = tf.Variable(np.random.uniform(low=-self.config['zeta_pos'],
                                                          high=self.config['zeta_pos'],
                                                          size=(ind_pnt_num, dim_x + dim_u)))
            self.zeta_mean = tf.Variable(self.config['zeta_mean'] * np.random.rand(ind_pnt_num, dim_x))
            zeta_var_unc = tf.Variable(backward(self.config['zeta_var'] * np.ones((ind_pnt_num, dim_x))))
            self.zeta_var = forward(zeta_var_unc)
            var_x_unc = tf.Variable(backward(self.config['var_x']))
            self.var_x = forward(var_x_unc)
            var_y_unc = tf.Variable(backward(self.config['var_y']))
            self.var_y = forward(var_y_unc)
            self.kern = RBF(self.config['gp_var'], self.config['gp_len'])
            self.var_dict = {'process noise': self.var_x,
                             'observation noise': self.var_y,
                             'kernel lengthscales': self.kern.lengthscales,
                             'kernel variance': self.kern.variance,
                             'IP pos': self.zeta_pos,
                             'IP mean': self.zeta_mean,
                             'IP var': self.zeta_var}

            # Loop init
            x_array = tf.TensorArray(dtype=tf.float64, size=self.seq_len_tf,
                                     clear_after_read=False)
            x_0 = self._recog_model(self.sample_in, self.sample_out)
            x_array = x_array.write(0, x_0)

            u_array = tf.TensorArray(dtype=tf.float64, size=self.seq_len_tf,
                                     clear_after_read=False)
            u_dub = tf.transpose(self.sample_in, perm=[1, 0, 2])
            u_dub = tf.tile(tf.expand_dims(u_dub, axis=2), [1, 1, samples, 1])
            u_array = u_array.unstack(u_dub)

            # Loop
            u_final, x_final, t_final = tf.while_loop(
                lambda u, x, t: t < self.seq_len_tf - 1,
                self._loop_body,
                [u_array, x_array, 0], parallel_iterations=1)

            x_final = tf.transpose(x_final.stack(), perm=[1, 0, 2, 3])
            self.y_final = x_final[:, :, :, :dim_y]

            # Likelihood
            var_y_exp = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.var_y, 0), 0), 0)
            var_full = tf.tile(var_y_exp, [self.batch_tf, self.seq_len_tf, samples, 1])
            y_dist = tf.contrib.distributions.MultivariateNormalDiag(
                loc=self.y_final, scale_diag=tf.sqrt(var_full))
            obs = tf.tile(tf.expand_dims(self.sample_out, 2), [1, 1, samples, 1])
            log_probs = y_dist.log_prob(obs)
            loglik = tf.reduce_sum(log_probs)

            # KL-Regularizer
            k_prior = self.kern.K(self.zeta_pos, self.zeta_pos)
            scale_prior = tf.tile(tf.expand_dims(tf.cholesky(k_prior), 0), [dim_x, 1, 1])
            zeta_prior = tf.contrib.distributions.MultivariateNormalTriL(
                loc=tf.zeros((dim_x, ind_pnt_num), dtype=tf.float64), scale_tril=scale_prior)
            zeta_dist = tf.contrib.distributions.MultivariateNormalDiag(
                loc=tf.transpose(self.zeta_mean), scale_diag=tf.sqrt(tf.transpose(self.zeta_var)))
            kl_reg = tf.reduce_sum(tf.contrib.distributions.kl_divergence(zeta_dist, zeta_prior))

            # Statistics
            self.pred_mean, self.pred_var = tf.nn.moments(self.y_final, axes=[2])
            self.pred_var = tf.add(self.pred_var, self.var_y)
            self.internal_mean, self.internal_var = tf.nn.moments(x_final, axes=[2])
            self.mse = tf.losses.mean_squared_error(labels=self.sample_out, predictions=self.pred_mean)
            self.sde = tf.abs(self.pred_mean - self.sample_out) / tf.sqrt(self.pred_var)

            # Training
            elbo = loglik * loss_factors[0] - kl_reg
            self.loss = tf.negative(elbo)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config['learning_rate'])
            self.train = optimizer.minimize(self.loss)
            self.saver = tf.train.Saver()
            self.init = tf.global_variables_initializer()

    def _loop_body(self, u, x, t):
        # config
        dim_u = self.config['ds'].dim_u
        dim_x = self.config['dim_x']
        samples = self.config['samples']

        # read input
        u_t = u.read(t)
        x_t = x.read(t)
        in_t = tf.concat((x_t, u_t), axis=2)

        # gp
        in_t_reshape = tf.reshape(in_t, (self.batch_tf * samples, dim_u + dim_x))

        fmean, fvar = conditional(in_t_reshape, self.zeta_pos, self.kern,
                                  self.zeta_mean, tf.sqrt(self.zeta_var))

        fmean = tf.reshape(fmean, (self.batch_tf, samples, dim_x))
        fvar = tf.reshape(fvar, (self.batch_tf, samples, dim_x))
        fmean = tf.add(fmean, in_t[:, :, :dim_x])
        fvar = fvar + self.var_x

        # sample
        eps = tf.tile(tf.random_normal((self.batch_tf, samples, 1), dtype=tf.float64), [1, 1, dim_x])
        x_t = tf.add(fmean, tf.multiply(eps, tf.sqrt(fvar)))
        x_out = x.write(t + 1, x_t)

        return u, x_out, t + 1

    def _recog_model(self, sample_in, sample_out):
        recog = self.config['recog_model']
        recog_len = self.config['recog_len']
        samples = self.config['samples']
        dim_x = self.config['dim_x']
        dim_y = self.config['ds'].dim_y
        x_0 = None

        if recog == 'output':
            x_0 = sample_out[:, 0, :]
            pad = tf.zeros((self.batch_tf, dim_x - dim_y), dtype=tf.float64)
            x_0 = tf.concat((x_0, pad), axis=1)
            x_0 = tf.tile(tf.expand_dims(x_0, axis=1), [1, samples, 1])

        if recog == 'conv':
            sample_uy = tf.concat((sample_in, sample_out), axis=2)
            sample_uy = sample_uy[:, :recog_len, :]
            sample_uy = tf.cast(sample_uy, tf.float32)

            layer1 = tf.layers.conv1d(sample_uy, 5, 3, activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling1d(layer1, 2, 2)
            out1 = tf.reshape(pool1, [self.batch_tf, 35])
            dense2 = tf.layers.dense(out1, dim_x)
            dense2 = tf.cast(dense2, tf.float64)

            x_0 = tf.expand_dims(dense2, axis=1) + tf.zeros((self.batch_tf, samples, dim_x), dtype=tf.float64)

        if recog == 'rnn':
            sample_uy = tf.concat((sample_in, sample_out), axis=2)
            sample_uy = sample_uy[:, :recog_len, :]
            rnn_recog = tf.nn.rnn_cell.GRUCell(16)
            initial_state = rnn_recog.zero_state(self.batch_tf, dtype=tf.float64)
            _, recog_state = tf.nn.dynamic_rnn(rnn_recog,
                                               tf.reverse(sample_uy, axis=[1]),
                                               initial_state=initial_state, dtype=tf.float64,
                                               scope='RNN_recog')
            dense = tf.layers.dense(recog_state, dim_x)
            x_0 = tf.tile(tf.expand_dims(dense, axis=1), [1, samples, 1])

        assert x_0 is not None, 'invalid config for recognition model'
        return x_0
