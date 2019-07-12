import numpy as np
from cbfssm.datasets.ds_manager import DSManager
from cbfssm.datasets.base_ds import BaseDS


class DSManagerDS(BaseDS):

    def __init__(self, seq_len, seq_stride):
        super(DSManagerDS, self).__init__(seq_len, seq_stride)

    def prepare_data(self, path, split, y_crop=None):
        # Load data
        u_data, _, y_data = DSManager.load_ds(path)
        if y_crop is not None:
            y_data = y_data[:, :y_crop]

        # Normalize
        self.normalize_init(u_data, y_data)
        u_data = self.normalize(u_data, 'in')
        y_data = self.normalize(y_data, 'out')

        # Save
        self.train_in = np.expand_dims(u_data[:split, :], axis=0)
        self.train_out = np.expand_dims(y_data[:split, :], axis=0)
        self.test_in = np.expand_dims(u_data[split:, :], axis=0)
        self.test_out = np.expand_dims(y_data[split:, :], axis=0)
        self.create_batches()


class RoboMoveSimple(DSManagerDS):

    dim_u = 2
    dim_y = 4

    def __init__(self, seq_len, seq_stride):
        super(RoboMoveSimple, self).__init__(seq_len, seq_stride)
        path = self.data_path + 'robomove_simple.mat'
        split = 25000
        self.prepare_data(path, split)


class RoboMove(DSManagerDS):

    dim_u = 2
    dim_y = 2

    def __init__(self, seq_len, seq_stride):
        super(RoboMove, self).__init__(seq_len, seq_stride)
        path = self.data_path + 'robomove.mat'
        split = 25000
        self.prepare_data(path, split)


class SpringNonlinear(DSManagerDS):

    dim_u = 1
    dim_y = 1

    def __init__(self, seq_len, seq_stride):
        super(SpringNonlinear, self).__init__(seq_len, seq_stride)
        path = self.data_path + 'spring_nonlinear.mat'
        split = 5000
        self.prepare_data(path, split, y_crop=1)
