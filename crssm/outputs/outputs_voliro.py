import math
import scipy.special
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from crssm.outputs.outputs import Outputs
from datasets.voliro_ext_ds import VoliroBaseDS
from sklearn.metrics import mean_squared_error
from matplotlib.lines import Line2D


class OutputsVoliro(Outputs):

    def __init__(self, *args):
        super(OutputsVoliro, self).__init__(*args)

    def _create_all(self, sess):
        self.training_stats()
        self.voliro_prediction(sess)
        self.var_dump(sess)
        self.error_dump(sess)

    def voliro_prediction(self, sess):
        model = self.model
        ds = self.ds

        # predict
        print("  voliro prediction")
        data_in = np.concatenate((ds.train_in[0:1, :, :], ds.test_in[0:1, :, :]), axis=1)
        data_out = np.concatenate((ds.train_out[0:1, :, :], ds.test_out[0:1, :, :]), axis=1)

        model.load_ds(sess, data_in, data_out)
        pred1, var1, ft1_pm, ft1_pred, ft1_var = sess.run(
            (model.pred_mean, model.pred_var, model.force_torque, model.ft_mean, model.ft_var))
        pred1, var1, ft1_pm, ft1_pred, ft1_var =\
            pred1[0, :, :], var1[0, :, :], ft1_pm[0, :, :], ft1_pred[0, :, :], ft1_var[0, :, :]
        eul1 = VoliroBaseDS.quat2eul(pred1[:, 3:7])
        pred1 = np.concatenate((pred1, eul1), axis=1)
        gt1 = data_out[0, :, :]

        model.load_ds(sess, ds.test_in2, ds.test_out2)
        pred2, var2, ft2_pm, ft2_pred, ft2_var = sess.run(
            (model.pred_mean, model.pred_var, model.force_torque, model.ft_mean, model.ft_var))
        pred2, var2, ft2_pm, ft2_pred, ft2_var =\
            pred2[0, :, :], var2[0, :, :], ft2_pm[0, :, :], ft2_pred[0, :, :], ft2_var[0, :, :]
        eul2 = VoliroBaseDS.quat2eul(pred2[:, 3:7])
        pred2 = np.concatenate((pred2, eul2), axis=1)
        gt2 = ds.test_out2[0, :, :]

        # # plot overview
        # pred_idx = [0, 13, 7, 10]
        # gt_idx = [0, 9, 3, 16]
        # plt.figure(1, figsize=(40, 14))
        # plt.suptitle('Flip and Tilt DS', fontsize=18)
        # for i, (predn, gtn) in enumerate([(pred1, gt1), (pred2, gt2)]):
        #     for k in range(4):
        #         plt.subplot(241 + k + 4*i)
        #         plt.plot(predn[:, 0+pred_idx[k]], 'r', label='pred x')
        #         plt.plot(predn[:, 1+pred_idx[k]], 'g', label='pred y')
        #         plt.plot(predn[:, 2+pred_idx[k]], 'b', label='pred z')
        #         plt.plot(gtn[:, 0+gt_idx[k]], 'r--', label='est x')
        #         plt.plot(gtn[:, 1+gt_idx[k]], 'g--', label='est y')
        #         plt.plot(gtn[:, 2+gt_idx[k]], 'b--', label='est z')
        #
        #         if i == 0:
        #             plt.axvline(x=ds.train_in.shape[1], color='k', linestyle='--')
        #         plt.legend()
        #         plt.title(['Position', 'Orientation', 'Lin. Velocity', 'Ang. Velocity'][k])
        #
        # plt.savefig(self.out_dir + '/voliro.pdf')
        # plt.close(1)

        # plot forces
        print("  voliro forces")
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

        # plot forces on test set
        print("  voliro forces test-set")
        plt.figure(2, figsize=(13.968*0.5/2.54, 13.968*0.5/12*8/2.54))

        plt.plot(ft2_pm[:, 0], 'r', label='pred x')
        plt.plot(gt2[:, 6], 'r--', label='est x')

        plt.plot(ft2_pm[:, 1], 'g', label='pred y')
        plt.plot(gt2[:, 7], 'g--', label='est y')

        plt.plot(ft2_pm[:, 2], 'b', label='pred z')
        plt.plot(gt2[:, 8], 'b--', label='est z')

        custom_lines = [Line2D([0], [0], color='r', lw=2, dashes=(0, 3, 20)),
                        Line2D([0], [0], color='g', lw=2, dashes=(0, 3, 20)),
                        Line2D([0], [0], color='b', lw=2, dashes=(0, 3, 20))]
        leg1 = plt.legend(custom_lines, ['x-force', 'y-force', 'z-force'], loc=2, ncol=3, mode='expand')
        custom_lines = [Line2D([0], [0], color='k', lw=2, dashes=(0, 3, 20)),
                        Line2D([0], [0], color='k', lw=2, dashes=(0, 1, 3, 1, 3, 1, 3))]
        plt.legend(custom_lines, ['prediction', 'ref'], loc=3, ncol=2, mode='expand')
        plt.gca().add_artist(leg1)

        plt.grid(True)
        plt.xlim([0, gt2.shape[0]])
        plt.ylim([-60, 50])

        plt.tight_layout(pad=0.2)
        plt.savefig(self.out_dir + '/voliro_prediction_test1.pdf')
        plt.close(2)

        plt.figure(2, figsize=(13.968*0.5/2.54, 13.968*0.5/12*8/2.54))

        plt.plot(ft2_pred[:, 0], 'r', label='pred x')
        lower = ft2_pred[:, 0] - 1.96 * np.sqrt(ft2_var[:, 0])
        upper = ft2_pred[:, 0] + 1.96 * np.sqrt(ft2_var[:, 0])
        plt.fill_between(range(ft2_pred.shape[0]), lower, upper, color=(1., 0.6, 0.6))
        plt.plot(gt2[:, 6], 'r--', label='est x')

        plt.plot(ft2_pred[:, 1], 'g', label='pred y')
        lower = ft2_pred[:, 1] - 1.96 * np.sqrt(ft2_var[:, 1])
        upper = ft2_pred[:, 1] + 1.96 * np.sqrt(ft2_var[:, 1])
        plt.fill_between(range(ft2_pred.shape[0]), lower, upper, color=(0.6, 1., 0.6))
        plt.plot(gt2[:, 7], 'g--', label='est y')

        plt.plot(ft2_pred[:, 2], 'b', label='pred z')
        lower = ft2_pred[:, 2] - 1.96 * np.sqrt(ft2_var[:, 2])
        upper = ft2_pred[:, 2] + 1.96 * np.sqrt(ft2_var[:, 2])
        plt.fill_between(range(ft2_pred.shape[0]), lower, upper, color=(0.6, 0.6, 1.))
        plt.plot(gt2[:, 8], 'b--', label='est z')

        plt.grid(True)
        plt.xlim([0, gt2.shape[0]])
        plt.ylim([-60, 50])

        plt.tight_layout(pad=0.2)
        plt.savefig(self.out_dir + '/voliro_prediction_test2.pdf')
        plt.close(2)

        # plot variance stats
        print("  voliro variance")
        train_size = ds.train_in[0, :, :].shape[0]
        sde1 = abs(ft1_pred[:, 0:3] - gt1[:, 6:9]) / np.sqrt(ft1_var[:, 0:3])
        sde_train = sde1[:train_size, :]
        sde_val = sde1[train_size:, :]
        sde_test = abs(ft2_pred[:, 0:3] - gt2[:, 6:9]) / np.sqrt(ft2_var[:, 0:3])

        plt.figure(3, figsize=(13.968 * 0.166 * 3 / 2.54, 13.968 * 0.11 * 3 / 2.54))
        var_max = 3
        x_axis = np.linspace(0, var_max, 1000)
        y_axis = np.asarray([(sde_train < x).sum() / (sde_train.shape[0] * 3) for x in x_axis])
        plt.plot(x_axis, y_axis, label='train')
        y_axis = np.asarray([(sde_val < x).sum() / (sde_val.shape[0] * 3) for x in x_axis])
        plt.plot(x_axis, y_axis, label='validate')
        y_axis = np.asarray([(sde_test < x).sum() / (sde_test.shape[0] * 3) for x in x_axis])
        plt.plot(x_axis, y_axis, label='test')
        error_y = scipy.special.erf(x_axis / math.sqrt(2))
        plt.plot(x_axis, error_y, 'k--', label='ref')
        plt.legend(loc=4)
        plt.xlabel('threshold (SD)')
        plt.ylabel('inliers (ratio)')
        plt.xlim([0, var_max])
        plt.ylim([0, 1])
        plt.grid(True)
        plt.tight_layout(pad=0.2)
        plt.savefig(self.out_dir + '/voliro_variance.pdf')
        plt.close(3)

    def error_dump(self, sess, filename='/error.txt'):
        model = self.model
        ds = self.ds

        # predict
        data_in = np.concatenate((ds.train_in[0:1, :, :], ds.test_in[0:1, :, :]), axis=1)
        data_out = np.concatenate((ds.train_out[0:1, :, :], ds.test_out[0:1, :, :]), axis=1)

        model.load_ds(sess, data_in, data_out)
        pred1, var1, ft1_pm, ft1_pred, ft1_var = sess.run(
            (model.pred_mean, model.pred_var, model.force_torque, model.ft_mean, model.ft_var))
        pred1, var1, ft1_pm, ft1_pred, ft1_var =\
            pred1[0, :, :], var1[0, :, :], ft1_pm[0, :, :], ft1_pred[0, :, :], ft1_var[0, :, :]
        eul1 = VoliroBaseDS.quat2eul(pred1[:, 3:7])
        pred1 = np.concatenate((pred1, eul1), axis=1)
        gt1 = data_out[0, :, :]

        model.load_ds(sess, ds.test_in2, ds.test_out2)
        pred2, var2, ft2_pm, ft2_pred, ft2_var = sess.run(
            (model.pred_mean, model.pred_var, model.force_torque, model.ft_mean, model.ft_var))
        pred2, var2, ft2_pm, ft2_pred, ft2_var =\
            pred2[0, :, :], var2[0, :, :], ft2_pm[0, :, :], ft2_pred[0, :, :], ft2_var[0, :, :]
        eul2 = VoliroBaseDS.quat2eul(pred2[:, 3:7])
        pred2 = np.concatenate((pred2, eul2), axis=1)
        gt2 = ds.test_out2[0, :, :]

        # dump
        train_len = ds.train_in.shape[1]
        vals = list()
        vals.append(np.sqrt(mean_squared_error(gt1[:train_len, 0:3], pred1[:train_len, 0:3])))
        vals.append(np.sqrt(mean_squared_error(gt1[:train_len, 3:6], pred1[:train_len, 7:10])))
        vals.append(np.sqrt(mean_squared_error(gt1[:train_len, 6:9], pred1[:train_len, 13:16])))
        vals.append(np.sqrt(mean_squared_error(gt1[train_len:, 0:3], pred1[train_len:, 0:3])))
        vals.append(np.sqrt(mean_squared_error(gt1[train_len:, 3:6], pred1[train_len:, 7:10])))
        vals.append(np.sqrt(mean_squared_error(gt1[train_len:, 6:9], pred1[train_len:, 13:16])))
        vals.append(np.sqrt(mean_squared_error(gt2[:, 0:3], pred2[:, 0:3])))
        vals.append(np.sqrt(mean_squared_error(gt2[:, 3:6], pred2[:, 7:10])))
        vals.append(np.sqrt(mean_squared_error(gt2[:, 6:9], pred2[:, 13:16])))
        vals.append(np.sqrt(mean_squared_error(gt1[:train_len, 6:9], ft1_pred[:train_len, 0:3])))
        vals.append(np.sqrt(mean_squared_error(gt1[train_len:, 6:9], ft1_pred[train_len:, 0:3])))
        vals.append(np.sqrt(mean_squared_error(gt2[:, 6:9], ft2_pred[:, 0:3])))
        np.savetxt(self.out_dir + filename, vals)
