import math
import numpy as np
import tensorflow as tf
from cbfssm.model.base_model import BaseModel
from cbfssm.utils.quaternions import Quaternion
from cbfssm.model.tf_transform import backward, forward
from cbfssm.model.gp_tf import GPModel


class Voliro(BaseModel):

    def __init__(self, config):
        self.gp_dim_in_f = 12
        self.gp_dim_out_f = 3
        self.gp_dim_in_b = 19
        self.gp_dim_out_b = 6
        self.dim_y = 7
        self.dim_x = 13
        super(Voliro, self).__init__(config)

    def _build_graph(self):
        with self.graph.as_default():
            self._setup_vars()
            self._local_coord()
            self._phys_model()
            self._gp_fun()
            self._io_arrays()
            self._backward()
            self._forward()
            self._build_loss()
            self._build_prediction()
            self._build_train()

    def _setup_vars(self):
        # physical quantities
        self.alloc_m = tf.constant(self._alloc_matrtix(), dtype=self.dtype)
        self.rotor_force_constant = tf.constant(0.000012, dtype=self.dtype)
        self.rotor_speed_max = tf.constant(1700, dtype=self.dtype)
        self.mass_inv = tf.constant(1./4.04, dtype=self.dtype)
        self.inertia_inv = tf.constant([1./0.078359127, 1./0.081797886, 1./0.1533554115], dtype=self.dtype)
        self.gravity = tf.constant([0., 0., 9.81], dtype=self.dtype)
        self.post_scale = self.rotor_force_constant * tf.pow(self.rotor_speed_max, 2.)
        # time
        timesteps = self.sample_in[0, :, 12]
        self.dt = (timesteps[-1] - timesteps[0]) / tf.cast(tf.shape(timesteps)[0], dtype=self.dtype)
        # noise
        self.var_x_unc = tf.Variable(backward(self.config['var_x']), dtype=self.dtype)
        self.var_x = forward(self.var_x_unc)
        self.var_y_unc = tf.Variable(backward(self.config['var_y']), dtype=self.dtype)
        self.var_y = forward(self.var_y_unc)
        self.var_z_unc = tf.Variable(backward(self.config['var_z']), dtype=self.dtype)
        self.var_z = forward(self.var_z_unc)
        # gp f
        self.gp_f = GPModel(in_dim=self.gp_dim_in_f,
                            out_dim=self.gp_dim_out_f,
                            num_points=self.config['ind_pnt_num'],
                            gp_var=self.config['gp_var'],
                            gp_len=self.config['gp_len'],
                            zeta_mean=self.config['zeta_mean'],
                            zeta_pos=self.config['zeta_pos'],
                            zeta_var=self.config['zeta_var'],
                            dtype=self.dtype)
        # gp b
        self.gp_b = GPModel(in_dim=self.gp_dim_in_b,
                            out_dim=self.gp_dim_out_b,
                            num_points=self.config['ind_pnt_num'],
                            gp_var=self.config['gp_var'],
                            gp_len=self.config['gp_len'],
                            zeta_mean=self.config['zeta_mean'],
                            zeta_pos=self.config['zeta_pos'],
                            zeta_var=self.config['zeta_var'],
                            dtype=self.dtype)
        # dump
        self.var_dict = {'process noise': self.var_x,
                         'observation noise': self.var_y,
                         'gp noise': self.var_z,
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

    def _local_coord(self):
        pwm, tilt = self.sample_in[:, :, :6], self.sample_in[:, :, 6:]
        local_coo = []
        for k in range(6):
            fac = tf.pow(pwm[..., k], 2)
            local_coo.append(tf.sin(tilt[..., k]) * fac)
            local_coo.append(tf.cos(tilt[..., k]) * fac)
        self.local_coo = tf.stack(local_coo, axis=-1)

    def _phys_model(self):
        data_in = self.local_coo
        shape = tf.shape(data_in)
        a_tile = tf.expand_dims(tf.expand_dims(self.alloc_m, axis=0), axis=0)
        a_tile = tf.tile(a_tile, [shape[0], shape[1], 1, 1])
        b_tile = tf.expand_dims(data_in, axis=-1)
        pred = tf.matmul(a_tile, b_tile)
        self.force_torque = tf.squeeze(pred, [3]) * self.post_scale

    def _gp_fun(self):
        samples = self.config['samples']
        # gp
        in_t = tf.reshape(self.local_coo, (self.batch_tf * self.seq_len_tf, self.gp_dim_in_f))
        fmean, fvar = self.gp_f.predict(in_t)
        fmean = tf.reshape(fmean, (self.batch_tf, self.seq_len_tf, self.gp_dim_out_f))
        fvar = tf.reshape(fvar, (self.batch_tf, self.seq_len_tf, self.gp_dim_out_f))
        fmean = tf.add(fmean, self.force_torque[..., :3])
        out_mean = tf.concat((fmean, self.force_torque[..., 3:]), axis=2)
        out_var = tf.concat((fvar, tf.zeros_like(self.force_torque[..., 3:])), axis=2)
        out_var += self.var_z
        # sampling
        out_mean_samp = tf.tile(tf.expand_dims(out_mean, axis=2), [1, 1, samples, 1])
        out_var_samp = tf.tile(tf.expand_dims(out_var, axis=2), [1, 1, samples, 1])
        eps = tf.tile(tf.random_normal((self.batch_tf, self.seq_len_tf, samples, 1), dtype=self.dtype),
                      [1, 1, 1, 6])
        self.ft_gp = tf.add(out_mean_samp, tf.multiply(eps, tf.sqrt(out_var_samp)))
        self.ft_mean, self.ft_var = out_mean, out_var

    def _io_arrays(self):
        samples = self.config['samples']

        y_array = tf.TensorArray(dtype=self.dtype, size=self.seq_len_tf,
                                 clear_after_read=False)
        y_dub = tf.transpose(self.out_to_hidden(self.sample_out), perm=[1, 0, 2])
        y_dub = tf.tile(tf.expand_dims(y_dub, axis=2), [1, 1, samples, 1])
        self.y_array = y_array.unstack(y_dub)

        u_dub = tf.transpose(self.ft_gp, [1, 0, 2, 3])
        u_array = tf.TensorArray(dtype=self.dtype, size=self.seq_len_tf,
                                 clear_after_read=False)
        self.u_array = u_array.unstack(u_dub)

    def _backward(self):
        samples = self.config['samples']

        prob_array = tf.TensorArray(dtype=self.dtype, size=self.seq_len_tf,
                                    clear_after_read=False)
        y2_array = tf.TensorArray(dtype=self.dtype, size=self.seq_len_tf,
                                  clear_after_read=False)

        y_init = tf.zeros((self.batch_tf, samples, self.gp_dim_out_b), dtype=self.dtype)
        u_final, y_final, y2_final, p_final, t_final, h_final = tf.while_loop(
            lambda u, y, y2, p, t, h: t >= 0,
            self._backward_body,
            [self.u_array, self.y_array, y2_array, prob_array, self.seq_len_tf - 1, y_init],
            parallel_iterations=1)

        y2_array = tf.transpose(y2_final.stack(), perm=[1, 0, 2, 3])
        out_dub = tf.tile(tf.expand_dims(self.out_to_hidden(self.sample_out), axis=2), [1, 1, samples, 1])
        self.y_tilde = tf.concat((out_dub, y2_array), axis=3)

        self.entropy = tf.reduce_sum(p_final.stack())

    def _backward_body(self, u, y, y2, p, t, h):
        samples = self.config['samples']

        # read input
        u_t = u.read(t)
        y_t = y.read(t)
        in_t = tf.concat((h, u_t, y_t), axis=2)

        # gp
        in_t_reshape = tf.reshape(in_t, (self.batch_tf * samples, self.gp_dim_in_b))
        fmean, fvar = self.gp_b.predict(in_t_reshape)

        fmean = tf.reshape(fmean, (self.batch_tf, samples, self.gp_dim_out_b))
        fvar = tf.reshape(fvar, (self.batch_tf, samples, self.gp_dim_out_b))
        fmean = tf.add(fmean, in_t[:, :, :self.gp_dim_out_b])

        # sampling
        eps = tf.tile(tf.random_normal((self.batch_tf, samples, 1), dtype=self.dtype), [1, 1, self.gp_dim_out_b])
        out = tf.add(fmean, tf.multiply(eps, tf.sqrt(fvar)))
        y2_out = y2.write(t, out)

        # entropy regularizer
        c = 2. * np.pi * np.e
        entropy = 0.5 * tf.reduce_sum(tf.log(c * fvar))
        p_out = p.write(t, entropy)

        return u, y, y2_out, p_out, t - 1, out

    def _forward(self):
        prob_array = tf.TensorArray(dtype=self.dtype, size=self.seq_len_tf - 1,
                                    clear_after_read=False)

        x_array = tf.TensorArray(dtype=self.dtype, size=self.seq_len_tf,
                                 clear_after_read=False)
        x_0 = self.y_tilde[:, 0, :, :]
        x_array = x_array.write(0, x_0)

        y_tilde_array = tf.TensorArray(dtype=self.dtype, size=self.seq_len_tf,
                                       clear_after_read=False)
        y_dub = tf.transpose(self.y_tilde, perm=[1, 0, 2, 3])
        y_tilde_array = y_tilde_array.unstack(y_dub)

        u_final, x_final, y_final, p_final, t_final = tf.while_loop(
            lambda u, x, y, p, t: t < self.seq_len_tf - 1,
            self._forward_body,
            [self.u_array, x_array, y_tilde_array, prob_array, 0], parallel_iterations=1)

        self.x_final = tf.transpose(x_final.stack(), perm=[1, 0, 2, 3])
        self.y_final = self.x_final[:, :, :, :self.dim_y]
        self.kl_x = tf.reduce_sum(p_final.stack())

    def _forward_body(self, u, x, y, p, t):
        samples = self.config['samples']

        # read input
        u_t = u.read(t)
        x_t = x.read(t)
        y_t = y.read(t + 1)

        # ode
        fmean = self.symplectic_euler(x_t, u_t)
        fvar = self.var_x
        fvar = tf.tile(tf.expand_dims(tf.expand_dims(fvar, axis=0), axis=0), [self.batch_tf, samples, 1])

        # sample q(x_t | x_{t-1}, y_t:T)
        var_y_tiled = tf.tile(tf.expand_dims(tf.expand_dims(self.var_y, axis=0), axis=0),
                              [self.batch_tf, samples, 1])
        y_diff = y_t - fmean
        s = var_y_tiled + fvar
        k = fvar * tf.reciprocal(s)
        mu = fmean + k * y_diff
        sig = tf.ones((self.batch_tf, samples, self.dim_x), dtype=self.dtype) - k
        sig = tf.square(sig) * fvar + tf.square(k) * var_y_tiled
        eps = tf.tile(tf.random_normal((self.batch_tf, samples, 1), dtype=self.dtype), [1, 1, self.dim_x])
        x_t = tf.add(mu, tf.multiply(eps, tf.sqrt(sig)))
        x_out = x.write(t + 1, x_t)

        # KL div regularizer x
        kl_reg = tf.log(fvar) - tf.log(sig) + (sig + tf.pow(mu - fmean, 2.)) / fvar - 1.
        kl_reg = 0.5 * tf.reduce_sum(kl_reg)
        p_out = p.write(t, kl_reg)

        return u, x_out, y, p_out, t + 1

    def _build_loss(self):
        samples = self.config['samples']

        # likelihood
        var_y_exp = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.var_y[:self.dim_y], 0), 0), 0)
        var_full = tf.tile(var_y_exp, [self.batch_tf, self.seq_len_tf, samples, 1])
        y_dist = tf.contrib.distributions.MultivariateNormalDiag(
            loc=self.y_final, scale_diag=tf.sqrt(var_full))
        obs = self.out_to_hidden(self.sample_out)
        obs = tf.tile(tf.expand_dims(obs, 2), [1, 1, samples, 1])
        self.log_probs = y_dist.log_prob(obs)
        self.loglik = tf.reduce_sum(self.log_probs)

        # KL div regularizer for z_f and z_b
        self.kl_z_f = self.gp_f.prior_kl()
        self.kl_z_b = self.gp_b.prior_kl()

        # prior on noise
        n_alpha_tf = tf.constant(self.config['n_beta'][0], dtype=self.dtype)
        n_beta_tf = tf.constant(self.config['n_beta'][1], dtype=self.dtype)
        n_dist = tf.distributions.Beta(n_alpha_tf, n_beta_tf)
        self.n_reg = tf.reduce_sum(n_dist.log_prob(self.var_z / self.config['n_beta'][2]))

        # prior on lengthscale
        l_alpha_tf = tf.constant(self.config['l_beta'][0], dtype=self.dtype)
        l_beta_tf = tf.constant(self.config['l_beta'][1], dtype=self.dtype)
        l_dist = tf.distributions.Beta(l_alpha_tf, l_beta_tf)
        self.l_reg = tf.reduce_sum(l_dist.log_prob(self.gp_f.kern.lengthscales / self.config['l_beta'][2]))

    def _build_prediction(self):
        self.pred_mean, self.pred_var = tf.nn.moments(self.x_final, axes=[2])
        self.pred_var += self.var_y

    def _build_train(self):
        loglik_factor = self.config['loglik_factor']
        samples = self.config['samples']

        divisor = tf.reciprocal(tf.constant(samples, dtype=self.dtype))
        elbo = (self.loglik * loglik_factor[0] * divisor
                - self.kl_x * loglik_factor[0] * divisor
                + self.entropy * loglik_factor[1] * divisor
                + self.n_reg * loglik_factor[2]
                + self.l_reg * loglik_factor[2]
                - self.kl_z_f - self.kl_z_b)
        self.loss = tf.negative(elbo)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config['learning_rate'])
        self.train = self.optimizer.minimize(self.loss)
        self.saver = tf.train.Saver()
        self.init = tf.global_variables_initializer()

    @staticmethod
    def _alloc_matrtix():
        angles = np.asarray([0.5, -0.5, -1. / 6., 5. / 6., 1. / 6., 7. / 6.]) * math.pi
        arm_length = 0.3
        a = np.zeros((6, 12))
        for i in range(6):
            a[0, 2 * i] = -math.cos(angles[i])
            a[0, 2 * i + 1] = 0
            a[1, 2 * i] = -math.sin(angles[i])
            a[1, 2 * i + 1] = 0
            a[2, 2 * i] = 0
            a[2, 2 * i + 1] = -1
            a[3, 2 * i] = 0
            a[3, 2 * i + 1] = -arm_length * math.cos(angles[i])
            a[4, 2 * i] = 0
            a[4, 2 * i + 1] = -arm_length * math.sin(angles[i])
            a[5, 2 * i] = -arm_length
            a[5, 2 * i + 1] = 0
        return a

    def symplectic_euler(self, y, force_torque):
        # read input
        pos = y[..., 0:3]
        rot = y[..., 3:7]
        linvel = y[..., 7:10]
        angvel = y[..., 10:13]

        # transform force and torque to global system
        f_glob = Quaternion.rot_vec(force_torque[..., :3], rot)
        t_glob = Quaternion.rot_vec(self.inertia_inv * force_torque[..., 3:], rot)

        # update velocities
        linvel_diff = self.mass_inv * f_glob
        linvel_diff += self.gravity
        angvel_diff = t_glob
        linvel += linvel_diff * self.dt
        angvel += angvel_diff * self.dt

        # update position and orientation
        rot_diff = 0.5 * Quaternion.multiply(Quaternion.pad_to_quat(angvel), rot)
        pos += linvel * self.dt
        rot += rot_diff * self.dt
        rot /= tf.tile(tf.expand_dims(tf.norm(rot, axis=-1), axis=-1), [1, 1, 4])

        return tf.concat((pos, rot, linvel, angvel), axis=-1)

    @staticmethod
    def out_to_hidden(y):
        return tf.concat((y[..., 0:3],
                          y[..., 12:16]), axis=-1)
