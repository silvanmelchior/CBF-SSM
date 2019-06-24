import numpy as np
from cbfssm.datasets.dsmanager_ds import RoboMove
from cbfssm.training.trainer import Trainer
from cbfssm.outputs.outputs_robomove import OutputsRoboMove
from cbfssm.model.cbfssm import CBFSSM


# curriculum learning scheme presented in appendix
# first train w/o entropy, then add it
for phase in range(2):

    #
    # Config
    #
    root_dir = "run_output/robomove"
    # dataset
    ds_sel = RoboMove
    seq_len = 300
    seq_stride = 50
    # model
    model_sel = CBFSSM
    dim_x = 4
    model_config = {
        # dataset
        'ds': ds_sel,
        'batch_size': 32,
        'shuffle': 10000,
        # method
        'dim_x': dim_x,
        'ind_pnt_num': 100,
        'samples': 50,
        'learning_rate': 0.01,
        'loss_factors': np.asarray([20., 2. * (phase == 1)]),
        'k_factor': 1.,
        'recog_len': 50,
        # variables init state
        'zeta_pos': 2.,
        'zeta_mean': 0.1 ** 2,
        'zeta_var': 0.01 ** 2,
        'var_x': np.asarray([0.1 ** 2] * dim_x),
        'var_y': np.asarray([1. ** 2] * dim_x),
        'gp_var': 0.1 ** 2,
        'gp_len': 1.
    }
    # training
    train = True
    retrain = (phase == 1)
    epochs = 100
    # evaluation
    output_sel = OutputsRoboMove


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
        trainer.train(ds, epochs, retrain=retrain)
        outputs.set_trainer(trainer)
    # evaluate
    outputs.create_all()
