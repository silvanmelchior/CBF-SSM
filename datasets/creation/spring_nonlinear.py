import math
import numpy as np
from datasets.linear_ds import LinearDS
from datasets.ds_manager import DSManager
import matplotlib.pyplot as plt


#
# DS
#
class SpringNonlinear(LinearDS):
    def propagate(self, u):
        u = np.tanh(u*2)
        super(SpringNonlinear, self).propagate(u)


#
# Config
#
ds_size = 10000
plot_size = 2000
save_ds = False
b = 0.05
k = 1.
m = 0.002
dt = 0.01
start = 1
sigma_x = 0
sigma_y = 1e-4
path = "datasets/data/spring_nonlinear.mat"
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
if save_ds:
    DSManager.save_ds(path, u_all, x_all, y_all, title)
    print("Saved " + title)
print("Plotting {pl} from {ds} points".format(pl=plot_size, ds=ds_size))


#
# Plot
#
plt.plot([p[0] for p in x_all][:plot_size], label='pos')
# plt.plot([p[1] for p in x_all][:plot_size], label='speed')
# plt.plot([p[2] for p in x_all][:plot_size], label='acc')
plt.plot(u_all[:plot_size], label='u')
# plt.plot(y_all[:plot_size], label='meas')
plt.legend()
plt.show()
