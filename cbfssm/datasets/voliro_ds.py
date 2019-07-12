import numpy as np
from cbfssm.datasets.base_ds import BaseDS
from cbfssm.datasets.voliro_loader import VoliroBaseDS


class VoliroDS(BaseDS):

    dim_u = 13
    dim_y = 22

    def __init__(self, seq_len, seq_stride):
        super(VoliroDS, self).__init__(seq_len, seq_stride)
        # Load
        mass = 4.04
        inertia = np.asarray([0.078359127, 0.081797886, 0.1533554115])
        ds1 = VoliroBaseDS(self.data_path + "voliro_tilt.mat", 1500, 3800)
        time1 = ds1.timesteps[:, None]
        battery1 = ds1.battery[:, None]
        u_data1 = np.concatenate((ds1.pwmup, ds1.tilt, time1), axis=1)
        y_data1 = np.concatenate((ds1.pos, ds1.linvel, ds1.linacc * mass,
                                  ds1.rpy, ds1.wxyz, ds1.angvel, ds1.angacc * inertia), axis=1)

        ds2 = VoliroBaseDS(self.data_path + "voliro_flip.mat", 17600, 20172)
        time2 = ds2.timesteps[:, None]
        battery2 = ds2.battery[:, None]
        u_data2 = np.concatenate((ds2.pwmup, ds2.tilt, time2), axis=1)
        y_data2 = np.concatenate((ds2.pos, ds2.linvel, ds2.linacc * mass,
                                  ds2.rpy, ds2.wxyz, ds2.angvel, ds2.angacc * inertia), axis=1)

        # Battery influence on PWM
        pwm_scale = np.sqrt(39.622609152 / 36.3063891724)
        battery_scale = battery2[0, 0]
        u_data1[:, :6] *= battery1 * pwm_scale / battery_scale
        u_data2[:, :6] *= battery2 * pwm_scale / battery_scale

        # Skip normalization
        self.mean['in'] = np.zeros(self.dim_u)
        self.std['in'] = np.ones(self.dim_u)
        self.mean['out'] = np.zeros(self.dim_y)
        self.std['out'] = np.ones(self.dim_y)

        # Save
        self._save(u_data1, y_data1, u_data2, y_data2)

        # Skip last sequence (because of 0-padding)
        self.train_in_batch = self.train_in_batch[:-1, :, :]
        self.train_out_batch = self.train_out_batch[:-1, :, :]
        self.test_in_batch = self.test_in_batch[:-1, :, :]
        self.test_out_batch = self.test_out_batch[:-1, :, :]

    def _save(self, u_data1, y_data1, u_data2, y_data2):
        pass


class VoliroTiltDS(VoliroDS):

    def __init__(self, seq_len, seq_stride):
        super(VoliroTiltDS, self).__init__(seq_len, seq_stride)

    def _save(self, u_data1, y_data1, u_data2, y_data2):
        split = int(u_data1.shape[0]/2)
        self.train_in = np.expand_dims(u_data1[:split, :], axis=0)
        self.train_out = np.expand_dims(y_data1[:split, :], axis=0)
        self.test_in = np.expand_dims(u_data1[split:, :], axis=0)
        self.test_out = np.expand_dims(y_data1[split:, :], axis=0)
        self.test_in2 = np.expand_dims(u_data2, axis=0)
        self.test_out2 = np.expand_dims(y_data2, axis=0)
        self.create_batches()


class VoliroFlipDS(VoliroDS):

    def __init__(self, seq_len, seq_stride):
        super(VoliroFlipDS, self).__init__(seq_len, seq_stride)

    def _save(self, u_data1, y_data1, u_data2, y_data2):
        split = int(u_data2.shape[0]/2)
        self.train_in = np.expand_dims(u_data2[:split, :], axis=0)
        self.train_out = np.expand_dims(y_data2[:split, :], axis=0)
        self.test_in = np.expand_dims(u_data2[split:, :], axis=0)
        self.test_out = np.expand_dims(y_data2[split:, :], axis=0)
        self.test_in2 = np.expand_dims(u_data1, axis=0)
        self.test_out2 = np.expand_dims(y_data1, axis=0)
        self.create_batches()
