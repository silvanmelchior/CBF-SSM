import numpy as np
import tensorflow as tf
from cbfssm.model.tf_transform import backward, forward
from cbfssm.model.gp_tf import GPModel
from cbfssm.model.base_model import BaseModel

import functools


class CBFSSM(BaseModel):

    def __init__(self, config, dtype=tf.float64):
        super(CBFSSM, self).__init__(config, dtype=dtype)

    def _build_graph(self):
        with self.graph.as_default():
            self._setup_vars()
            self._io_arrays()
            self._backward()
            self._forward()
            self._build_loss()
            self._build_prediction()
            self._build_train()

    def _setup_vars(self):
        dim_u = self.config['ds'].dim_u
        dim_x = self.config['dim_x']
        dim_y = self.config['ds'].dim_y

        self.gp_f = GPModel(in_dim=dim_x + dim_u,
                            out_dim=dim_x,
                            num_points=self.config['ind_pnt_num'],
                            gp_var=self.config['gp_var'],
                            gp_len=self.config['gp_len'],
                            zeta_mean=self.config['zeta_mean'],
                            zeta_pos=self.config['zeta_pos'],
                            zeta_var=self.config['zeta_var'],
                            dtype=self.dtype)

        self.gp_b = GPModel(in_dim=dim_x + dim_u,
                            out_dim=dim_x - dim_y,
                            num_points=self.config['ind_pnt_num'],
                            gp_var=self.config['gp_var'],
                            gp_len=self.config['gp_len'],
                            zeta_mean=self.config['zeta_mean'],
                            zeta_pos=self.config['zeta_pos'],
                            zeta_var=self.config['zeta_var'],
                            dtype=self.dtype)

        # Observation and process noise
        self.var_x_unc = tf.Variable(backward(self.config['var_x']), dtype=self.dtype)
        self.var_x = forward(self.var_x_unc)
        self.var_y_unc = tf.Variable(backward(self.config['var_y']), dtype=self.dtype)
        self.var_y = forward(self.var_y_unc)

        self.var_dict = {'process noise': self.var_x,
                         'observation noise': self.var_y,
                         'kernel lengthscales f': self.gp_f.kern.lengthscales,
                         'kernel variance f': self.gp_f.kern.variance,
                         'IP pos f': self.gp_f.zeta_pos,
                         'IP mean f': self.gp_f.zeta_mean,
                         'IP var f': self.gp_f.zeta_var,
                         'kernel lengthscales b': self.gp_b.kern.lengthscales,
                         'kernel variance b': self.gp_b.kern.variance,
                         'IP pos b': self.gp_b.zeta_pos,
                         'IP mean b': self.gp_b.zeta_mean,
                         'IP var b': self.gp_b.zeta_var}

    def _io_arrays(self):
        samples = self.config['samples']

        self.u_array = tf.TensorArray(dtype=self.dtype, size=self.seq_len_tf,
                                      clear_after_read=False)
        u_dub = tf.transpose(self.sample_in, perm=[1, 0, 2])
        u_dub = tf.tile(tf.expand_dims(u_dub, axis=2), [1, 1, samples, 1])
        self.u_array = self.u_array.unstack(u_dub)

        self.y_array = tf.TensorArray(dtype=self.dtype, size=self.seq_len_tf,
                                      clear_after_read=False)
        y_dub = tf.transpose(self.sample_out, perm=[1, 0, 2])
        y_dub = tf.tile(tf.expand_dims(y_dub, axis=2), [1, 1, samples, 1])
        self.y_array = self.y_array.unstack(y_dub)

    def _backward(self):
        samples = self.config['samples']

        prob_array = tf.TensorArray(dtype=self.dtype, size=self.seq_len_tf,
                                    clear_after_read=False)
        y2_array = tf.TensorArray(dtype=self.dtype, size=self.seq_len_tf,
                                  clear_after_read=False)

        y2_array, prob_array = self._backward_run(y2_array, prob_array, 0)
        y2_array, prob_array = self._backward_run(y2_array, prob_array, 1)

        y2_array = tf.transpose(y2_array.stack(), perm=[1, 0, 2, 3])
        out_dub = tf.tile(tf.expand_dims(self.sample_out, axis=2), [1, 1, samples, 1])
        self.y_tilde = tf.concat((out_dub, y2_array), axis=3)

        self.entropy = tf.reduce_sum(prob_array.stack())

    def _backward_run(self, y2_array, prob_array, run):
        samples = self.config['samples']
        dim_x = self.config['dim_x']
        dim_y = self.config['ds'].dim_y

        y_init = tf.zeros((self.batch_tf, samples, dim_x - dim_y), dtype=self.dtype)
        u_final, y_final, y2_final, p_final, t_final, h_final = tf.while_loop(
            lambda u, y, y2, p, t, h: t >= 0,
            functools.partial(self._backward_body, run=run),
            [self.u_array, self.y_array, y2_array, prob_array, self.seq_len_tf - 1, y_init],
            parallel_iterations=1)
        return y2_final, p_final

    def _backward_body(self, u, y, y2, p, t, h, run):
        dim_u = self.config['ds'].dim_u
        dim_x = self.config['dim_x']
        dim_y = self.config['ds'].dim_y
        dim_out = dim_x - dim_y
        samples = self.config['samples']
        recog_len = self.config['recog_len']

        # conditions
        if run == 0:
            resample_cond = tf.equal(tf.mod(t + 1, 2*recog_len), 0)
            write_cond = tf.mod(t, 2*recog_len) < recog_len
        else:
            resample_cond = tf.equal(tf.mod(t + recog_len + 1, 2*recog_len), 0)
            write_cond = tf.mod(t, 2*recog_len) >= recog_len

        # read input
        u_t = u.read(t)
        y_t = y.read(t)
        hidden = tf.cond(resample_cond,
                         lambda: tf.tile(tf.random_normal((self.batch_tf, samples, 1), dtype=self.dtype),
                                         [1, 1, dim_out]),
                         lambda: h)
        in_t = tf.concat((hidden, u_t, y_t), axis=2)

        # gp
        in_t_reshape = tf.reshape(in_t, (self.batch_tf * samples, dim_x + dim_u))
        fmean, fvar = self.gp_b.predict(in_t_reshape)

        fmean = tf.reshape(fmean, (self.batch_tf, samples, dim_out))
        fvar = tf.reshape(fvar, (self.batch_tf, samples, dim_out))
        fmean = tf.add(fmean, in_t[:, :, :dim_out])
        fvar = fvar + self.var_x[:dim_out]

        # sampling
        # p_next = tfp.distributions.MultivariateNormalDiag(fmean, fvar)
        # out = p_next.sample(1).squeeze(0)
        eps = tf.tile(tf.random_normal((self.batch_tf, samples, 1), dtype=self.dtype), [1, 1, dim_out])
        out = tf.add(fmean, tf.multiply(eps, tf.sqrt(fvar)))
        y2_out = tf.cond(write_cond, lambda: y2.write(t, out), lambda: y2)

        # entropy regularizer
        c = tf.constant(2. * np.pi * np.e, dtype=self.dtype)
        entropy = 0.5 * tf.reduce_sum(tf.log(c) + tf.log(fvar))
        p_out = tf.cond(write_cond, lambda: p.write(t, entropy), lambda: p)

        return u, y, y2_out, p_out, t - 1, out

    def _forward(self):
        dim_y = self.config['ds'].dim_y

        prob_array = tf.TensorArray(dtype=self.dtype, size=self.seq_len_tf - 1,
                                    clear_after_read=False)

        x_array = tf.TensorArray(dtype=self.dtype, size=self.seq_len_tf,
                                 clear_after_read=False)
        self.x_0 = self.y_tilde[:, 0, :, :]
        x_array = x_array.write(0, self.x_0)

        y_tilde_array = tf.TensorArray(dtype=self.dtype, size=self.seq_len_tf,
                                       clear_after_read=False)
        y_dub = tf.transpose(self.y_tilde, perm=[1, 0, 2, 3])
        y_tilde_array = y_tilde_array.unstack(y_dub)

        u_final, x_final, y_final, p_final, t_final = tf.while_loop(
            lambda u, x, y, p, t: t < self.seq_len_tf - 1,
            self._forward_body,
            [self.u_array, x_array, y_tilde_array, prob_array, 0], parallel_iterations=1)

        self.x_final = tf.transpose(x_final.stack(), perm=[1, 0, 2, 3])
        self.y_final = self.x_final[:, :, :, :dim_y]
        self.kl_x = tf.reduce_sum(p_final.stack())

    def _forward_body(self, u, x, y, p, t):
        # config
        dim_u = self.config['ds'].dim_u
        dim_x = self.config['dim_x']
        samples = self.config['samples']
        recog_len = self.config['recog_len']
        k_factor = self.config['k_factor']

        # read input
        u_t = u.read(t)
        x_t = x.read(t)
        y_t = y.read(t + 1)
        in_t = tf.concat((x_t, u_t), axis=2)

        # gp
        in_t_reshape = tf.reshape(in_t, (self.batch_tf * samples, dim_u + dim_x))
        fmean, fvar = self.gp_f.predict(in_t_reshape)

        fmean = tf.reshape(fmean, (self.batch_tf, samples, dim_x))
        fvar = tf.reshape(fvar, (self.batch_tf, samples, dim_x))
        fmean = tf.add(fmean, in_t[:, :, :dim_x])
        fvar = fvar + self.var_x

        # sampling randomness
        eps = tf.tile(tf.random_normal((self.batch_tf, samples, 1), dtype=self.dtype), [1, 1, dim_x])

        # sample q(x_t | x_{t-1}, y_t:T)
        var_y_tiled = tf.tile(tf.expand_dims(tf.expand_dims(self.var_y, axis=0), axis=0),
                              [self.batch_tf, samples, 1])
        var_y_tiled += (k_factor - 1.) * fvar
        y_diff = y_t - fmean
        s = var_y_tiled + fvar
        k = fvar * tf.reciprocal(s)
        mu = fmean + k * y_diff
        sig = tf.ones((self.batch_tf, samples, dim_x), dtype=self.dtype) - k
        sig = tf.square(sig) * fvar + tf.square(k) * var_y_tiled
        x_t = tf.add(mu, tf.multiply(eps, tf.sqrt(sig)))

        # sample p(x_t | x_{t-1})
        x_t_nocond = tf.add(fmean, tf.multiply(eps, tf.sqrt(fvar)))

        # choose correct sample
        do_cond = tf.logical_or(self.condition, t < recog_len - 1)
        x_next = tf.cond(do_cond, lambda: x_t, lambda: x_t_nocond)
        x_out = x.write(t + 1, x_next)

        # KL div regularizer x
        kl_reg = tf.log(fvar) - tf.log(sig) + (sig + tf.pow(mu - fmean, 2.)) / fvar - 1.
        kl_reg = tf.reduce_sum(kl_reg) * tf.cond(
            do_cond, lambda: tf.constant(0.5, dtype=self.dtype), lambda: tf.constant(0., dtype=self.dtype))
        p_out = p.write(t, kl_reg)

        return u, x_out, y, p_out, t + 1

    def _build_loss(self):
        dim_y = self.config['ds'].dim_y
        samples = self.config['samples']
        loss_factors = self.config['loss_factors']

        # likelihood
        var_y_exp = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.var_y[:dim_y], 0), 0), 0)
        var_full = tf.tile(var_y_exp, [self.batch_tf, self.seq_len_tf, samples, 1])
        y_dist = tf.contrib.distributions.MultivariateNormalDiag(
            loc=self.y_final, scale_diag=tf.sqrt(var_full))
        obs = tf.tile(tf.expand_dims(self.sample_out, 2), [1, 1, samples, 1])
        log_probs = y_dist.log_prob(obs)
        loglik = tf.reduce_sum(log_probs)

        # KL div regularizer for z_f and z_b
        kl_z_f = self.gp_f.prior_kl()
        kl_z_b = self.gp_b.prior_kl()

        divisor = tf.reciprocal(tf.constant(samples, dtype=self.dtype))
        elbo = (loglik * loss_factors[0] * divisor
                - self.kl_x * loss_factors[0] * divisor
                + self.entropy * loss_factors[1] * divisor
                - kl_z_f - kl_z_b)
        self.loss = tf.negative(elbo)

    def _build_prediction(self):
        dim_y = self.config['ds'].dim_y

        self.pred_mean, self.pred_var = tf.nn.moments(self.y_final, axes=[2])
        self.pred_var = tf.add(self.pred_var, self.var_y[:dim_y])
        self.internal_mean, self.internal_var = tf.nn.moments(self.x_final, axes=[2])
        self.mse = tf.losses.mean_squared_error(labels=self.sample_out, predictions=self.pred_mean)
        self.sde = tf.abs(self.pred_mean - self.sample_out) / tf.sqrt(self.pred_var)

    def _build_train(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config['learning_rate'])
        self.train = optimizer.minimize(self.loss)
        self.saver = tf.train.Saver()
        self.init = tf.global_variables_initializer()
