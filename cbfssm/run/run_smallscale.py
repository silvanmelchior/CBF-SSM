import sys
import math
import numpy as np
from cbfssm.datasets.prssm_ds import Actuator, Ballbeam, Drive, Furnace, Dryer
from cbfssm.training.trainer import Trainer
from cbfssm.outputs.outputs import Outputs
from cbfssm.outputs.output_summary import OutputSummary
from cbfssm.model.cbfssm import CBFSSM


# Choose Tasks
datasets = [(Actuator, 'actuator', 0.01, 100),
            (Ballbeam, 'ballbeam', 0.001, 10),
            (Drive, 'drive', 0.01, 50),
            (Dryer, 'dryer', 0.003, 100),
            (Furnace, 'furnace', 0.003, 100)]
tasks = [int(sys.argv[1])] if len(sys.argv) > 1 else range(len(datasets))


# Run
for task_nr in tasks:

    #
    # Config
    #
    root_dir = "run_output/smallscale/" + datasets[task_nr][1]
    ds_sel = datasets[task_nr][0]
    model_sel = CBFSSM
    output_sel = Outputs
    iterations = 5
    train = True
    retrain = False
    seq_len = 50
    seq_stride = 1
    dim_x = 4
    ind_pnt_num = 20
    batch_size = 10
    datapoints = 3000 * batch_size
    samples = 50
    learning_rate = 0.1
    loss_factors = np.asarray([1., 1., 0.]) * datasets[task_nr][2]
    cond_factor = [1., datasets[task_nr][3]]
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
            'var_y': np.asarray([1. ** 2] * ds_sel.dim_y),
            'gp_var': 0.5 ** 2,
            'gp_len': 2.
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
            epochs = math.ceil(datapoints/ds.train_in_batch.shape[0])
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
