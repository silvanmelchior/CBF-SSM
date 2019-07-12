import math
import numpy as np
from cbfssm.datasets.ds_manager import DSManager


#
# DS
#
class RoboMoveDS:
    """Dataset that simulates a robot moving in 2d teritory"""

    def __init__(self, start_pos, start_orient, sigma_x, sigma_y):
        assert(len(start_pos.shape) == 1)
        assert(start_pos.shape[0] == 2)
        assert(np.isscalar(start_orient))
        self.pos = start_pos
        self.orient = start_orient
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def get_state(self):
        return np.concatenate((self.pos, [self.orient]))

    def propagate_fn(self, x, u):
        """x is interpreted as [pos_x, pos_y, orientation].
        u is interpreted as [speed, curvature].
        speed corresponds to the distance traveled in this step,
        curvature is the reciprocal of the radius"""
        pos = np.asarray([x[0], x[1]], dtype=np.float64)
        orient = x[2]
        speed = u[0]
        curv = u[1]
        orient_x = math.sin(orient)
        orient_y = math.cos(orient)
        if abs(curv) < 1e-5:
            pos += np.asarray([orient_x, orient_y]) * speed
        else:
            sign = np.sign(curv)
            normal = np.asarray([orient_y, -orient_x]) * sign
            radius = 1. / abs(curv)
            angle = (speed / radius) * sign
            angle_c, angle_s = np.cos(angle), np.sin(angle)
            j = np.matrix([[angle_c, angle_s], [-angle_s, angle_c]])
            normal_rot = np.dot(j, normal)
            normal_rot = np.squeeze(np.asarray(normal_rot))
            pos += (normal - normal_rot)*radius
            orient += angle
        cov = np.eye(2) * self.sigma_x
        pos += rand_arr(2, cov)
        while orient < 0:
            orient += 2.*math.pi
        orient = orient % (2.*math.pi)
        return np.concatenate((pos, [orient]))

    def evaluate_propagate(self, x, u):
        """propagate function for particle filter, uses np matrices for x"""
        x_in = np.squeeze(np.asarray(x.T))
        x_new = self.propagate_fn(x_in, u)
        return np.matrix(x_new).T

    def propagate(self, u):
        """propagate dataset with input u"""
        x = self.propagate_fn(self.get_state(), u)
        self.pos = np.asarray([x[0], x[1]])
        self.orient = x[2]

    def measure(self):
        """measure position of robot"""
        cov = np.eye(2) * self.sigma_y
        return self.pos + rand_arr(2, cov)

    @staticmethod
    def get_xdim():
        """get dimensionality of hidden state"""
        return 3


class RoboMoveSimpleDS:
    """As above, but full observation and continuous internal representation"""

    def __init__(self, start_pos, start_orient, sigma_x, sigma_y):
        assert(len(start_pos.shape) == 1)
        assert(start_pos.shape[0] == 2)
        assert(np.isscalar(start_orient))
        self.pos = start_pos
        self.orient = np.asarray([math.sin(start_orient), math.cos(start_orient)])
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def get_state(self):
        return np.concatenate((self.pos, self.orient))

    def propagate_fn(self, x, u):
        """x is interpreted as [pos_x, pos_y, orientation].
        u is interpreted as [speed, curvature].
        speed corresponds to the distance traveled in this step,
        curvature is the reciprocal of the radius"""
        pos = np.asarray([x[0], x[1]], dtype=np.float64)
        scale = math.sqrt(x[2]**2 + x[3]**2)
        orient_x = x[2] / scale
        orient_y = x[3] / scale
        orient = np.asarray([orient_x, orient_y])
        speed = u[0]
        curv = u[1]
        if abs(curv) < 1e-5:
            pos += np.asarray([orient_x, orient_y]) * speed
        else:
            sign = np.sign(curv)
            normal = np.asarray([orient_y, -orient_x]) * sign
            radius = 1. / abs(curv)
            angle = (speed / radius) * sign
            angle_c, angle_s = np.cos(angle), np.sin(angle)
            j = np.matrix([[angle_c, angle_s], [-angle_s, angle_c]])
            normal_rot = np.dot(j, normal)
            normal_rot = np.squeeze(np.asarray(normal_rot))
            pos += (normal - normal_rot)*radius
            orient = np.dot(j, orient)
            orient = np.squeeze(np.asarray(orient))
        cov = np.eye(2) * self.sigma_x
        pos += rand_arr(2, cov)
        return np.concatenate((pos, orient))

    def evaluate_propagate(self, x, u):
        """propagate function for particle filter, uses np matrices for x"""
        x_in = np.squeeze(np.asarray(x.T))
        x_new = self.propagate_fn(x_in, u)
        return np.matrix(x_new).T

    def propagate(self, u):
        """propagate dataset with input u"""
        x = self.propagate_fn(self.get_state(), u)
        self.pos = np.asarray([x[0], x[1]])
        self.orient = np.asarray([x[2], x[3]])

    def measure(self):
        """measure position of robot"""
        cov = np.eye(4) * self.sigma_y
        return self.get_state() + rand_arr(4, cov)

    @staticmethod
    def get_xdim():
        """get dimensionality of hidden state"""
        return 4


def rand_arr(length, cov):
    if length == 0:
        return []
    else:
        return np.random.multivariate_normal([0.]*length, cov)


#
# Config
#
ds_size = 30000
sigma_x = 1e-5
sigma_y = 1e-4
start_pos = np.asarray([0., 0.])
start_orient = 0.
u_state = 0
u_val = [0, 0]
u_ts = 0
path = "robomove_simple.mat"
title = 'RoboMoveSimple-sx{sx}-sy{sy}'.format(sx=sigma_x, sy=sigma_y)


def u_default(_):
    speed = np.random.uniform(-0.1, 0.5)
    if speed < 0:
        speed = 0
    if np.random.binomial(1, 0.3):
        curv = 0
    else:
        curv = np.random.uniform(-1.5, 1.5)
    return np.asarray([speed, curv])


def u_fn(ts, x):
    global u_state, u_val, u_ts
    dist = math.sqrt(x[0]**2 + x[1]**2)
    if dist < 5.:
        u_state = 0
        return u_default(ts)
    else:
        if u_state == 0:
            u_state = 1
            u_ts = ts
            speed = np.random.uniform(0.2, 0.5)
            curv = np.random.uniform(0.5, 0.8)
            sign = np.random.binomial(1, 0.5)*2. - 1.
            u_val = np.asarray([speed, curv*sign])
        speed = u_val[0]
        slow_down = 1. / (ts - u_ts + 1)
        curv = 0.8*u_val[1] + 0.2*slow_down*u_val[1]
        return np.asarray([speed, curv])


#
# Sample DS
#
ds = RoboMoveSimpleDS(start_pos, start_orient, sigma_x, sigma_y)
u_all, x_all, y_all = DSManager.sample_ds(ds, ds_size, u_fn)
DSManager.save_ds(path, u_all, x_all, y_all, title)
print("Saved " + title)
