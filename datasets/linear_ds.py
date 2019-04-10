import numpy as np
from utils.rand_vec import rand_vec


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
        self.x = self.A*self.x + self.B*u + rand_vec(len(self.x), self.Q)
    
    def measure(self):
        return self.C*self.x + rand_vec(self.R.shape[0], self.R)
