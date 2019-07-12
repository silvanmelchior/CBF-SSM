import numpy as np
from cbfssm.datasets import RoboMove
from cbfssm.training import Trainer
from cbfssm.outputs import Outputs
from cbfssm.model import CBFSSM


#
# Config
#
root_dir = "run_output/my_own_experiment"
# dataset
ds_sel = RoboMove     # set to your new dataset class
seq_len = 100         # length of sub-trajectories for training
seq_stride = 50       # distance between two sub-trajectories, e.g. 100 would result in no overlap (as seq_len=100),
# model
model_sel = CBFSSM    # use CBFSSMHALF here if you have no unstable hidden dimension (e.g. no hidden dimensions at all)
dim_x = 4             # dimensionality of latent state
model_config = {
    # dataset
    'ds': ds_sel,
    'batch_size': 32,
    'shuffle': 10000,                       # shuffle buffer size
    # method
    'dim_x': dim_x,
    'ind_pnt_num': 100,                     # number of inducing points
    'samples': 50,                          # number of particles
    'learning_rate': 0.01,
    'loss_factors': np.asarray([10., 0.]),  # lambdas in paper. Start with large lambda_1 and no entropy (lambda_2=0)
    'k_factor': 1.,                         # $k$-factor in paper, start with e.g. 50 if have stable dataset
    'recog_len': 50,                        # 2*t' in paper, number of steps for recognition model
    # variables init state (can leave as-is in most cases)
    'zeta_pos': 2.,
    'zeta_mean': 0.1 ** 2,
    'zeta_var': 0.01 ** 2,
    'var_x': np.asarray([0.1 ** 2] * dim_x),
    'var_y': np.asarray([1. ** 2] * dim_x),  # change dim_x to ds_sel.dim_y for CBFSSMHALF
    'gp_var': 0.1 ** 2,
    'gp_len': 1.
}
# training
train = True          # set to False if have already trained and want to re-evaluate only
epochs = 100          # number of epochs for training (make sure in training plot afterwards that converged)
# evaluation
output_sel = Outputs  # can create new class deriving from it if need richer outputs


#
# Run (can leave as-is in most cases)
#     (if need multiple runs, can use class OutputSummary as helper, see e.g. run_sarcos.py)
#
# load
outputs = output_sel(root_dir)
ds = ds_sel(seq_len, seq_stride)
outputs.set_ds(ds)
model = model_sel(model_config)
outputs.set_model(model, root_dir)
# train
if train:
    trainer = Trainer(model, root_dir)
    trainer.train(ds, epochs)
    outputs.set_trainer(trainer)
# evaluate
outputs.create_all()
