"""
Parts of this code were provided by Karen Bodie, 2018
"""

import numpy as np
import scipy.io
from utils.hom_transform import euler_from_quaternion, euler_matrix, quaternion_from_euler
from utils.quaternions import Quaternion


class VoliroBaseDS:

    def __init__(self, ds_path, startidx, endidx):

        # load data
        ds_raw = scipy.io.loadmat(ds_path)['dataset']
        ds = {}
        for key in ['TIME_StartTime', 'LPOS_X', 'LPOS_Y', 'LPOS_Z', 'LPSP_X', 'LPSP_Y', 'LPSP_Z',
                    'ATT_qw', 'ATT_qx', 'ATT_qy', 'ATT_qz',
                    'ATSP_qw', 'ATSP_qx', 'ATSP_qy', 'ATSP_qz',
                    'OUT0_Out2', 'OUT0_Out3', 'OUT0_Out4', 'OUT0_Out5', 'OUT0_Out6', 'OUT0_Out7',
                    'OUT1_Out0', 'OUT1_Out1', 'OUT1_Out2', 'OUT1_Out3', 'OUT1_Out4', 'OUT1_Out5',
                    'ATC0_Out0', 'ATC0_Out1', 'ATC0_Out2', 'ATC0_Out3', 'ATC0_Out4', 'ATC0_Out5',
                    'ATC1_Out0', 'ATC1_Out1', 'ATC1_Out2', 'ATC1_Out3', 'ATC1_Out4', 'ATC1_Out5',
                    'ATC2_Out0', 'ATC2_Out1', 'ATC2_Out2', 'ATC2_Out3', 'ATC2_Out4', 'ATC2_Out5',
                    'BATT_VFilt']:
            ds[key] = self._process_array(ds_raw[key])

        # position
        x_est = ds['LPOS_X'][startidx:endidx]
        y_est = ds['LPOS_Y'][startidx:endidx]
        z_est = ds['LPOS_Z'][startidx:endidx]
        x_est -= x_est[0]
        y_est -= y_est[0]
        z_est -= z_est[0]
        self.pos = np.stack((x_est, y_est, z_est)).T

        # attitude
        qw_est = ds['ATT_qw'][startidx:endidx]
        qx_est = ds['ATT_qx'][startidx:endidx]
        qy_est = ds['ATT_qy'][startidx:endidx]
        qz_est = ds['ATT_qz'][startidx:endidx]
        self.wxyz = np.stack((qw_est, qx_est, qy_est, qz_est)).T
        self.rpy = self.quat2eul(self.wxyz)
        self.wxyz = np.asarray([quaternion_from_euler(self.rpy[i, 0], self.rpy[i, 1], self.rpy[i, 2], axes='rxyz')
                                for i in range(self.rpy.shape[0])])

        # pwm
        out_0_0 = ds['ATC0_Out0'][startidx:endidx]
        out_0_1 = ds['ATC0_Out1'][startidx:endidx]
        out_0_2 = ds['ATC0_Out2'][startidx:endidx]
        out_0_3 = ds['ATC0_Out3'][startidx:endidx]
        out_0_4 = ds['ATC0_Out4'][startidx:endidx]
        out_0_5 = ds['ATC0_Out5'][startidx:endidx]
        self.pwmup = np.stack((out_0_0, out_0_1, out_0_2, out_0_3, out_0_4, out_0_5)).T
        out_1_0 = ds['ATC1_Out0'][startidx:endidx]
        out_1_1 = ds['ATC1_Out1'][startidx:endidx]
        out_1_2 = ds['ATC1_Out2'][startidx:endidx]
        out_1_3 = ds['ATC1_Out3'][startidx:endidx]
        out_1_4 = ds['ATC1_Out4'][startidx:endidx]
        out_1_5 = ds['ATC1_Out5'][startidx:endidx]
        self.pwmlo = np.stack((out_1_0, out_1_1, out_1_2, out_1_3, out_1_4, out_1_5)).T

        # tilt
        atc_2_0 = np.rad2deg(ds['ATC2_Out0'][startidx:endidx])
        atc_2_1 = np.rad2deg(ds['ATC2_Out1'][startidx:endidx])
        atc_2_2 = np.rad2deg(ds['ATC2_Out2'][startidx:endidx])
        atc_2_3 = np.rad2deg(ds['ATC2_Out3'][startidx:endidx])
        atc_2_4 = np.rad2deg(ds['ATC2_Out4'][startidx:endidx])
        atc_2_5 = np.rad2deg(ds['ATC2_Out5'][startidx:endidx])
        self.tilt = np.deg2rad(np.stack((atc_2_0, atc_2_1, atc_2_2, atc_2_3, atc_2_4, atc_2_5)).T)

        # time
        self.dt = (ds['TIME_StartTime'][endidx] - ds['TIME_StartTime'][startidx]) / float((endidx-startidx)*1000000)
        self.timesteps = ds['TIME_StartTime'][startidx:endidx] / 1000000.

        # smooth
        sigma = 25
        self.pos_smooth = self.smooth_signal(self.pos, sigma)
        self.rpy_smooth = self.smooth_signal(self.rpy, sigma)
        self.wxyz_smooth = self.smooth_signal(self.wxyz, sigma)
        g = np.asarray([0., 0., -9.81])

        # linear velocity
        self.linvel = [np.zeros(3)]
        for i in range(1, self.pos_smooth.shape[0]):
            linmom = self.pos_smooth[i] - self.pos_smooth[i-1]
            linmom /= self.dt
            self.linvel.append(linmom)
        self.linvel = np.asarray(self.linvel)

        # linear acceleration
        self.linacc = [np.zeros(3)]
        for i in range(1, self.pos_smooth.shape[0] - 1):
            linacc = self.linvel[i + 1] - self.linvel[i]
            linacc /= self.dt
            self.linacc.append(linacc)
        self.linacc.append(np.zeros(3))
        self.linacc = np.asarray(self.linacc)
        for i in range(self.linacc.shape[0]):
            roll, pitch, yaw = self.rpy[i]
            g_rot = np.transpose(euler_matrix(roll, pitch, yaw, 'rxyz')[:3, :3]).dot(g)
            self.linacc[i] += g_rot

        # angular velocity
        self.angvel = [np.zeros(3)]
        for i in range(1, self.rpy_smooth.shape[0]):
            angmom = self.wxyz_smooth[i] - self.wxyz_smooth[i-1]
            angmom /= self.dt
            angmom = angmom[None, :]
            angmom = 2. * Quaternion.multiply_np(angmom, Quaternion.invert_np(self.wxyz_smooth[i][None, :]))
            angmom = angmom[0, 1:]
            self.angvel.append(angmom)
        self.angvel = np.asarray(self.angvel)

        # angular acceleration
        self.angacc = [np.zeros(3)]
        for i in range(1, self.rpy_smooth.shape[0] - 1):
            angacc = self.angvel[i + 1] - self.angvel[i]
            angacc /= self.dt
            self.angacc.append(angacc)
        self.angacc.append(np.zeros(3))
        self.angacc = np.asarray(self.angacc)

        # battery
        self.battery = np.asarray(ds['BATT_VFilt'][startidx:endidx]) / 25.

    @staticmethod
    def _process_array(array):
        return np.asarray(array[0][0]).T[0]

    @staticmethod
    def quat2eul(eul_array):
        rpy_est = np.asarray([euler_from_quaternion(eul_array[i, :], axes='rxyz')
                              for i in range(eul_array.shape[0])])
        rpy_est[:, 2] -= rpy_est[0, 2]
        rpy_est = VoliroBaseDS.filtereuleranglesdeg(rpy_est)
        return rpy_est

    @staticmethod
    def filtereuleranglesdeg(vector):
        threshold = 2./3. * np.pi
        vector = np.atleast_2d(vector)
        vector2 = np.zeros_like(vector)
        vector2[0, :] = vector[0, :]
        for k in range(vector.shape[1]):
            for i in range(1, vector.shape[0]):
                if vector[i, k] - vector2[i-1, k] > threshold:
                    vector2[i, k] = vector[i, k] - 2 * np.pi
                    if vector2[i, k] - vector2[i-1, k] > threshold:
                        vector2[i, k] = vector[i, k] - 2 * np.pi
                elif vector[i, k] - vector2[i-1, k] < -threshold:
                    vector2[i, k] = vector[i, k] + np.pi
                    if vector2[i, k] - vector2[i-1, k] < -threshold:
                        vector2[i, k] = vector[i, k] + 2 * np.pi
                else:
                    vector2[i, k] = vector[i, k]
        return vector2

    @staticmethod
    def smooth_signal(x, sigma):
        x_new = x.copy()
        for i in range(x.shape[1]):
            x_new[:, i] = scipy.ndimage.filters.gaussian_filter1d(x_new[:, i], sigma)
        return x_new
