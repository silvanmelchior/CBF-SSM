import numpy as np
from crssm.datasets.prssm_ds import Sarcos
from crssm.training.trainer import Trainer
from crssm.outputs.outputs import Outputs
from crssm.outputs.output_summary import OutputSummary
from crssm.model.crssm import CRSSM


#
# Config
#
root_dir = "run_output/sarcos"
ds_sel = Sarcos
model_sel = CRSSM
output_sel = Outputs
iterations = 5
epochs = 8
train = True
retrain = False
seq_len = 250
seq_stride = 10
dim_x = 14
ind_pnt_num = 100
batch_size = 5
samples = 20
learning_rate = 0.05
loss_factors = np.asarray([0.3, 0.3, 0.])
cond_factor = [1., 50.]
recog_len = 16


#
# Run
#
summary = OutputSummary(root_dir)
for it in range(iterations):

    # Iteration Config
    if iterations != 1:
        print("\n=== Iteration %d ===\n" % it)
    out_dir = root_dir if iterations == 1 else root_dir + "/run_%d" % it

    model_config = {
        # dataset
        'ds': ds_sel,
        'batch_size': batch_size,
        'shuffle': 10000,
        # method
        'dim_x': dim_x,
        'ind_pnt_num': ind_pnt_num,
        'samples': samples,
        'learning_rate': learning_rate,
        'loss_factors': loss_factors,
        'cond_factor': cond_factor,
        'recog_len': recog_len,
        # variables init state
        'zeta_pos': 2.,
        'zeta_mean': 0.05 ** 2,
        'zeta_var': 0.01 ** 2,
        'var_x': np.asarray([0.002 ** 2] * dim_x),
        'var_y': np.asarray([0.05 ** 2] * dim_x),
        'gp_var': 0.5 ** 2,
        'gp_len': 1.
    }

    # Prepare
    outputs = output_sel(out_dir)

    # Dataset
    ds = ds_sel(seq_len, seq_stride)
    outputs.set_ds(ds)

    # Model
    model = model_sel(model_config)
    outputs.set_model(model, out_dir)

    # Train
    if train:
        trainer = Trainer(model, out_dir)
        trainer.train(ds, epochs, retrain=retrain)
        outputs.set_trainer(trainer)

    # Test
    outputs.create_all()
    summary.add_outputs(outputs)


#
# Close up
#
summary.write_summary()
print('done')
