import numpy as np
from cbfssm.datasets.voliro_ds import VoliroFlipDS
from cbfssm.training.trainer import Trainer
from cbfssm.outputs.outputs_voliro import OutputsVoliro
from cbfssm.model.voliro import Voliro


#
# Config
#
root_dir = "run_output/voliro"
# dataset
ds_sel = VoliroFlipDS
seq_len = 64
seq_stride = 50
# model
model_sel = Voliro
model_config = {
    # dataset
    'ds': ds_sel,
    'batch_size': 16,
    'shuffle': 10000,
    # method
    'ind_pnt_num': 20,
    'samples': 20,
    'learning_rate': 0.01,
    'loglik_factor': np.asarray([20., 0., 0.2 * 20 * 50]),
    'n_beta': [10., 2., 10.],
    'l_beta': [1., 10., 10.],
    # variables init state
    'zeta_pos': 2.,
    'zeta_mean': 0.05 ** 2,
    'zeta_var': 0.01 ** 2,
    'gp_var': 0.5 ** 2,
    'gp_len': 5.,
    'var_x': np.asarray([0.02, 0.02, 0.02,
                         0.02, 0.02, 0.02, 0.02,
                         0.2, 0.2, 0.2,
                         0.2, 0.2, 0.2]) ** 2,
    'var_y': np.asarray([0.02, 0.02, 0.02,
                         0.02, 0.02, 0.02, 0.02,
                         0.2, 0.2, 0.2,
                         0.2, 0.2, 0.2]) ** 2,
    'var_z': np.asarray([0.02] * 6)
}
# training
train = True
epochs = 2000
# evaluation
output_sel = OutputsVoliro


#
# Run
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
