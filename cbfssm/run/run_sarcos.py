import numpy as np
from cbfssm.datasets.prssm_ds import Sarcos
from cbfssm.training.trainer import Trainer
from cbfssm.outputs.outputs import Outputs
from cbfssm.outputs.output_summary import OutputSummary
from cbfssm.model.cbfssm import CBFSSM


#
# Config
#
root_dir = "run_output/sarcos"
iterations = 5
# dataset
ds_sel = Sarcos
seq_len = 250
seq_stride = 10
# model
model_sel = CBFSSM
dim_x = 14
model_config = {
    # dataset
    'ds': ds_sel,
    'batch_size': 5,
    'shuffle': 10000,
    # method
    'dim_x': dim_x,
    'ind_pnt_num': 100,
    'samples': 20,
    'learning_rate': 0.05,
    'loss_factors': np.asarray([0.3, 0.]),
    'cond_factor': [1., 50.],
    'recog_len': 16,
    # variables init state
    'zeta_pos': 2.,
    'zeta_mean': 0.05 ** 2,
    'zeta_var': 0.01 ** 2,
    'var_x': np.asarray([0.002 ** 2] * dim_x),
    'var_y': np.asarray([0.05 ** 2] * dim_x),
    'gp_var': 0.5 ** 2,
    'gp_len': 1.
}
# training
train = True
epochs = 8
# evaluation
output_sel = Outputs


#
# Run
#
summary = OutputSummary(root_dir)
for it in range(iterations):

    # iteration Config
    if iterations != 1:
        print("\n=== Iteration %d ===\n" % it)
    out_dir = root_dir if iterations == 1 else root_dir + "/run_%d" % it
    # load
    outputs = output_sel(out_dir)
    ds = ds_sel(seq_len, seq_stride)
    outputs.set_ds(ds)
    model = model_sel(model_config)
    outputs.set_model(model, out_dir)
    # train
    if train:
        trainer = Trainer(model, out_dir)
        trainer.train(ds, epochs)
        outputs.set_trainer(trainer)
    # evaluate
    outputs.create_all()
    summary.add_outputs(outputs)


#
# Summarize
#
summary.write_summary()
