import math
import numpy as np
from utils.rand_vec import rand_arr


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

    def get_xdim(self):
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

    def get_xdim(self):
        """get dimensionality of hidden state"""
        return 4


class RoboMoveNoRotDS:
    """As above, but full observation and continuous internal representation"""

    def __init__(self, start_pos, sigma_x, sigma_y, contraction=0.7):
        assert(len(start_pos.shape) == 1)
        assert(start_pos.shape[0] == 2)
        self.pos = start_pos
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.contraction = contraction

    def get_state(self):
        return self.pos

    def propagate_fn(self, x, u):
        x_new = x + u
        x_new *= self.contraction
        cov = np.eye(2) * self.sigma_x
        x_new += rand_arr(2, cov)
        return x_new

    def propagate(self, u):
        x = self.propagate_fn(self.get_state(), u)
        self.pos = x

    def measure(self):
        cov = np.eye(2) * self.sigma_y
        return self.get_state() + rand_arr(2, cov)


def distort_view(x, theta=0.7*0.5*math.pi, move=11):
    x_new = x[0]
    y_new = x[1] * math.cos(theta)
    z_new = x[1] * math.sin(theta) + move
    y_new = y_new / z_new
    x_new = x_new / z_new
    return [x_new, y_new]
