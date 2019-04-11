import numpy as np
from crssm.datasets.base_ds import BaseDS
from datasets.prssm.real_world_tasks import SarcosArm
from datasets.prssm.real_world_tasks import Actuator as ActuatorBuilder
from datasets.prssm.real_world_tasks import Ballbeam as BallbeamBuilder
from datasets.prssm.real_world_tasks import Drive as DriveBuilder
from datasets.prssm.real_world_tasks import Gas_furnace as FurnaceBuilder
from datasets.prssm.real_world_tasks import Dryer as DryerBuilder


class PRSSMDS(BaseDS):

    def __init__(self, seq_len, seq_stride):
        super(PRSSMDS, self).__init__(seq_len, seq_stride)

    def prepare_data(self, ds_sel):
        # Load Dataset
        ds = ds_sel()
        ds.load_data()
        data_in = np.asarray(ds.data_in_train).reshape((-1, self.dim_u))
        data_out = np.asarray(ds.data_out_train).reshape((-1, self.dim_y))
        self.normalize_init(data_in, data_out)

        # Save
        self.train_in = self.normalize(np.asarray(ds.data_in_train), 'in')
        self.train_out = self.normalize(np.asarray(ds.data_out_train), 'out')
        self.test_in = self.normalize(np.asarray(ds.data_in_test), 'in')
        self.test_out = self.normalize(np.asarray(ds.data_out_test), 'out')
        self.create_batches()


class Sarcos(PRSSMDS):

    dim_u = 7
    dim_y = 7

    def __init__(self, seq_len, seq_stride):
        super(Sarcos, self).__init__(seq_len, seq_stride)
        self.prepare_data(SarcosArm)


class Actuator(PRSSMDS):

    dim_u = 1
    dim_y = 1

    def __init__(self, seq_len, seq_stride):
        super(Actuator, self).__init__(seq_len, seq_stride)
        self.prepare_data(ActuatorBuilder)


class Ballbeam(PRSSMDS):

    dim_u = 1
    dim_y = 1

    def __init__(self, seq_len, seq_stride):
        super(Ballbeam, self).__init__(seq_len, seq_stride)
        self.prepare_data(BallbeamBuilder)


class Drive(PRSSMDS):

    dim_u = 1
    dim_y = 1

    def __init__(self, seq_len, seq_stride):
        super(Drive, self).__init__(seq_len, seq_stride)
        self.prepare_data(DriveBuilder)


class Furnace(PRSSMDS):

    dim_u = 1
    dim_y = 1

    def __init__(self, seq_len, seq_stride):
        super(Furnace, self).__init__(seq_len, seq_stride)
        self.prepare_data(FurnaceBuilder)


class Dryer(PRSSMDS):

    dim_u = 1
    dim_y = 1

    def __init__(self, seq_len, seq_stride):
        super(Dryer, self).__init__(seq_len, seq_stride)
        self.prepare_data(DryerBuilder)
