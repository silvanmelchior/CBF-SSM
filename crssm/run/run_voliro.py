import numpy as np
from crssm.datasets.voliro_ds import VoliroFlipDS
from crssm.training.trainer import Trainer
from crssm.outputs.outputs_voliro import OutputsVoliro
from crssm.outputs.output_summary import OutputSummary
from crssm.model.voliro import Voliro


#
# Config
#
root_dir = "run_output/voliro"
ds_sel = VoliroFlipDS
model_sel = Voliro
output_sel = OutputsVoliro
cond_factor = 1
iterations = 1
train = True
retrain = False
epochs = 2000
seq_len = 64
seq_stride = 50
ind_pnt_num = 20
batch_size = 16
samples = 20
learning_rate = 0.01
loglik_factor = np.asarray([64. / seq_len, 0.2*20*50, 0.])
var_y = np.asarray([0.02, 0.02, 0.02,
                    0.02, 0.02, 0.02, 0.02,
                    0.2, 0.2, 0.2,
                    0.2, 0.2, 0.2]) ** 2
var_x = var_y
var_z = np.asarray([0.02] * 6)
n_beta = [10., 2., 10.]
l_beta = [1., 10., 10.]

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
        'cond_factor': cond_factor,
        'ind_pnt_num': ind_pnt_num,
        'samples': samples,
        'learning_rate': learning_rate,
        'loglik_factor': loglik_factor,
        'n_beta': n_beta,
        'l_beta': l_beta,
        # variables init state
        'zeta_pos': 2.,
        'zeta_mean': 0.05**2,
        'zeta_var': 0.01**2,
        'gp_var': 0.5**2,
        'gp_len': 0.5 * l_beta[2],
        'var_x': var_x,
        'var_y': var_y,
        'var_z': var_z
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
