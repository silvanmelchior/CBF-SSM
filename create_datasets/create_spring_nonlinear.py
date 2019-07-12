import math
import numpy as np
from cbfssm.datasets.ds_manager import DSManager


#
# DS
#
class LinearDS:

    def __init__(self, A, B, C, Q, R, x):
        """A: state-transition model
           B: control-input model
           C: observation model
           Q: transition noise covariance
           R: observation noise covariance
           x: start vector
           assumes all inputs are np-matrices"""
        self.A = np.matrix(A)
        self.B = np.matrix(B)
        self.C = np.matrix(C)
        self.Q = np.matrix(Q)
        self.R = np.matrix(R)
        self.x = np.matrix(x)

    def get_state(self):
        return self.x

    def propagate(self, u):
        self.x = self.A * self.x + self.B * u + rand_vec(len(self.x), self.Q)

    def measure(self):
        return self.C * self.x + rand_vec(self.R.shape[0], self.R)


class SpringNonlinear(LinearDS):
    def propagate(self, u):
        u = np.tanh(u*2)
        super(SpringNonlinear, self).propagate(u)


def rand_vec(dim, cov):
    r = np.random.multivariate_normal([0] * dim, cov)
    r = np.matrix.transpose(np.matrix(r))
    return r


#
# Config
#
ds_size = 10000
b = 0.05
k = 1.
m = 0.002
dt = 0.01
start = 1
sigma_x = 0
sigma_y = 1e-4
path = "spring_nonlinear.mat"
title = 'Spring-Nonlinear-b{b}-k{k}-m{m}-dt{dt}-sx{sx}-sy{sy}-u_randint'.format(
    b=b, k=k, m=m, dt=dt, sx=sigma_x, sy=sigma_y)
rand_int = np.random.uniform(low=-2, high=2, size=math.floor(ds_size/100))


def u_fn(ts, _):
    return np.matrix('1') * rand_int[math.floor(ts / ds_size * len(rand_int))]


#
# Create DS
#
A = np.matrix([[1., dt, 0.], [0., 1., dt], [-k/m, -b/m, 0.]])
B = np.matrix([[0.], [0.], [1./m]])
C = np.matrix('1,0,0')
Q = np.eye(3) * sigma_x
R = np.eye(1) * sigma_y
x_0 = np.matrix('1;0;0') * start

ds = SpringNonlinear(A, B, C, Q, R, x_0)
for _ in range(5):
    ds.propagate(u_fn(0, 0))

u_all, x_all, y_all = DSManager.sample_ds_matrix(ds, ds_size, u_fn)
DSManager.save_ds(path, u_all, x_all, y_all, title)
print("Saved " + title)
