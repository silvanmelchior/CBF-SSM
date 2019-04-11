import math
import numpy as np
from datasets.robomove_ds import RoboMoveSimpleDS
from datasets.ds_manager import DSManager


#
# Config
#
ds_size = 30000
save_ds = False
sigma_x = 1e-5
sigma_y = 1e-4
start_pos = np.asarray([0., 0.])
start_orient = 0.
u_state = 0
u_val = [0, 0]
u_ts = 0
path = "datasets/data/robomove_simple.mat"
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

if save_ds:
    DSManager.save_ds(path, u_all, x_all, y_all, title)
    print("Saved " + title)
