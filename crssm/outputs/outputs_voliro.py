import numpy as np
import matplotlib.pyplot as plt
from crssm.outputs.outputs import Outputs
from matplotlib.lines import Line2D


class OutputsVoliro(Outputs):

    def __init__(self, *args):
        super(OutputsVoliro, self).__init__(*args)

    def _create_all(self, sess):
        self.training_stats()
        self.voliro_forces(sess)
        self.var_dump(sess)

    def voliro_forces(self, sess):
        model = self.model
        ds = self.ds

        # predict
        print("  voliro forces")
        data_in = np.concatenate((ds.train_in[0:1, :, :], ds.test_in[0:1, :, :]), axis=1)
        data_out = np.concatenate((ds.train_out[0:1, :, :], ds.test_out[0:1, :, :]), axis=1)

        model.load_ds(sess, data_in, data_out)
        pred1, var1, ft1_pm, ft1_pred, ft1_var = sess.run(
            (model.pred_mean, model.pred_var, model.force_torque, model.ft_mean, model.ft_var))
        pred1, var1, ft1_pm, ft1_pred, ft1_var =\
            pred1[0, :, :], var1[0, :, :], ft1_pm[0, :, :], ft1_pred[0, :, :], ft1_var[0, :, :]
        gt1 = data_out[0, :, :]

        model.load_ds(sess, ds.test_in2, ds.test_out2)
        pred2, var2, ft2_pm, ft2_pred, ft2_var = sess.run(
            (model.pred_mean, model.pred_var, model.force_torque, model.ft_mean, model.ft_var))
        pred2, var2, ft2_pm, ft2_pred, ft2_var =\
            pred2[0, :, :], var2[0, :, :], ft2_pm[0, :, :], ft2_pred[0, :, :], ft2_var[0, :, :]
        gt2 = ds.test_out2[0, :, :]

        # plot forces
        fig = plt.figure(2, figsize=(13.968*0.9/2.54, 13.968*0.9/12*8/2.54))

        for i, (predn, gtn) in enumerate([(ft1_pm, gt1), (ft2_pm, gt2)]):
            ax = fig.add_subplot(221 + i)

            plt.plot(predn[:, 0], 'r', label='pred x')
            plt.plot(gtn[:, 6], 'r--', label='est x')

            plt.plot(predn[:, 1], 'g', label='pred y')
            plt.plot(gtn[:, 7], 'g--', label='est y')

            plt.plot(predn[:, 2], 'b', label='pred z')
            plt.plot(gtn[:, 8], 'b--', label='est z')

            if i == 0:
                plt.ylabel('Physical Model')

            if i == 1:
                custom_lines = [Line2D([0], [0], color='r', lw=2),
                                Line2D([0], [0], color='g', lw=2),
                                Line2D([0], [0], color='b', lw=2)]
                leg1 = ax.legend(custom_lines, ['x-force', 'y-force', 'z-force'], loc=4)
                custom_lines = [Line2D([0], [0], color='k', lw=2),
                                Line2D([0], [0], color='k', linestyle='--', lw=2)]
                ax.legend(custom_lines, ['prediction', 'ref'], loc=3)
                ax.add_artist(leg1)

            plt.grid(True)
            plt.xlim([0, gtn.shape[0]])

        for i, (predn, varn, gtn) in enumerate([(ft1_pred, ft1_var, gt1), (ft2_pred, ft2_var, gt2)]):
            plt.subplot(223 + i)

            plt.plot(predn[:, 0], 'r', label='pred x')
            lower = predn[:, 0] - 1.96 * np.sqrt(varn[:, 0])
            upper = predn[:, 0] + 1.96 * np.sqrt(varn[:, 0])
            plt.fill_between(range(predn.shape[0]), lower, upper, color=(1., 0.6, 0.6))
            plt.plot(gtn[:, 6], 'r--', label='est x')

            plt.plot(predn[:, 1], 'g', label='pred y')
            lower = predn[:, 1] - 1.96 * np.sqrt(varn[:, 1])
            upper = predn[:, 1] + 1.96 * np.sqrt(varn[:, 1])
            plt.fill_between(range(predn.shape[0]), lower, upper, color=(0.6, 1., 0.6))
            plt.plot(gtn[:, 7], 'g--', label='est y')

            plt.plot(predn[:, 2], 'b', label='pred z')
            lower = predn[:, 2] - 1.96 * np.sqrt(varn[:, 2])
            upper = predn[:, 2] + 1.96 * np.sqrt(varn[:, 2])
            plt.fill_between(range(predn.shape[0]), lower, upper, color=(0.6, 0.6, 1.))
            plt.plot(gtn[:, 8], 'b--', label='est z')

            if i == 0:
                plt.axvline(x=ds.train_in.shape[1], color='k', linestyle='--')
                plt.title('Train, Validate')
                plt.ylabel('Physical Model + CR-SSM')
            else:
                plt.title('Test')

            plt.grid(True)
            plt.xlim([0, gtn.shape[0]])

        plt.tight_layout(pad=0.2)
        plt.savefig(self.out_dir + '/voliro_prediction.pdf')
        plt.close(2)
