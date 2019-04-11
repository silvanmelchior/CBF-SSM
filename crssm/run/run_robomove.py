import numpy as np
from crssm.datasets.dsmanager_ds import RoboMove
from crssm.training.trainer import Trainer
from crssm.outputs.outputs_robomove import OutputsRoboMove
from crssm.outputs.output_summary import OutputSummary
from crssm.model.crssm import CRSSM


for phase in range(2):

    #
    # Config
    #
    root_dir = "run_output/robomove"
    ds_sel = RoboMove
    model_sel = CRSSM
    output_sel = OutputsRoboMove
    iterations = 1
    train = True
    retrain = (phase == 1)
    epochs = 100
    seq_len = 300
    seq_stride = 50
    dim_x = 4
    ind_pnt_num = 100
    batch_size = 32
    samples = 50
    learning_rate = 0.01
    loss_factors = np.asarray([1., 1., 0.2 * (phase == 1)]) * 20. * 32 * 300 / float(samples * batch_size * seq_len)
    cond_factor = [1., 1.]
    recog_len = 50

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
            'recog_model': 'rnn',
            # variables init state
            'zeta_pos': 2.,
            'zeta_mean': 0.1 ** 2,
            'zeta_var': 0.01 ** 2,
            'var_x': np.asarray([0.1 ** 2] * dim_x),
            'var_y': np.asarray([1. ** 2] * dim_x),
            'gp_var': 0.1 ** 2,
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
