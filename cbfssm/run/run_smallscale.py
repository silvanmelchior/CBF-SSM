import sys
import math
import numpy as np
from cbfssm.datasets.prssm_ds import Actuator, Ballbeam, Drive, Furnace, Dryer
from cbfssm.training.trainer import Trainer
from cbfssm.outputs.outputs import Outputs
from cbfssm.outputs.output_summary import OutputSummary
from cbfssm.model.cbfssm import CBFSSM


# Choose Tasks
datasets = [(Actuator, 'actuator', 0.01,  100),
            (Ballbeam, 'ballbeam', 0.001, 10),
            (Drive,    'drive',    0.01,  50),
            (Dryer,    'dryer',    0.003, 100),
            (Furnace,  'furnace',  0.003, 100)]
tasks = [int(sys.argv[1])] if len(sys.argv) > 1 else range(len(datasets))


# Execute Tasks
for task_nr in tasks:

    #
    # Config
    #
    root_dir = "run_output/smallscale/" + datasets[task_nr][1]
    iterations = 5
    # dataset
    ds_sel = datasets[task_nr][0]
    seq_len = 50
    seq_stride = 1
    # model
    model_sel = CBFSSM
    dim_x = 4
    model_config = {
        # dataset
        'ds': ds_sel,
        'batch_size': 10,
        'shuffle': 10000,
        # method
        'dim_x': dim_x,
        'ind_pnt_num': 20,
        'samples': 50,
        'learning_rate': 0.1,
        'loss_factors': np.asarray([1., 0.]) * datasets[task_nr][2],
        'cond_factor': [1., datasets[task_nr][3]],
        'recog_len': 16,
        # variables init state
        'zeta_pos': 2.,
        'zeta_mean': 0.05 ** 2,
        'zeta_var': 0.01 ** 2,
        'var_x': np.asarray([0.002 ** 2] * dim_x),
        'var_y': np.asarray([1. ** 2] * dim_x),  # TODO: was ds_sel.dim_y before
        'gp_var': 0.5 ** 2,
        'gp_len': 2.
    }
    # training
    train = True
    train_iterations = 30000
    # evaluation
    output_sel = Outputs

    #
    # Run
    #
    summary = OutputSummary(root_dir)
    for it in range(iterations):

        # iteration config
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
            epochs = math.ceil(train_iterations/ds.train_in_batch.shape[0])
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
