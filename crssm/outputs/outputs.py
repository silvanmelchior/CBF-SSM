import os
import math
import sklearn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.special
import scipy.io


class Outputs:

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.ds = None
        self.model = None
        self.model_path = None
        self.trainer = None
        self.last_rmse = None
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def set_ds(self, ds):
        self.ds = ds

    def set_model(self, model, model_dir):
        self.model = model
        self.model_path = model_dir + '/best.ckpt'

    def set_trainer(self, trainer):
        self.trainer = trainer

    def get_last_rmse(self):
        return self.last_rmse

    def create_all(self):
        assert self.model is not None
        assert self.ds is not None
        with self.model.graph.as_default():
            with tf.Session() as sess:
                self.model.saver.restore(sess, self.model_path)
                print("Generating outputs...")
                self._create_all(sess)

    def _create_all(self, sess):
        self.training_stats()
        self.prediction(sess)
        self.test_mse(sess)
        self.var_dump(sess)

    def training_stats(self):
        if self.trainer is not None:
            print("  training stats")
            plt.figure(1)
            plt.plot(self.trainer.train_all, label='train')
            plt.plot(self.trainer.test_all, label='test')
            plt.legend()
            plt.savefig(self.out_dir + '/training_loss.pdf')
            plt.close(1)

    def prediction(self, sess, predict_size=300):
        print("  prediction")
        model = self.model
        ds = self.ds
        predict_size = min(ds.train_in.shape[1], predict_size)

        # Train
        model.load_ds(sess, ds.train_in[0:1, :predict_size, :],
                      ds.train_out[0:1, :predict_size, :])
        pred_train, var_train = sess.run((model.pred_mean, model.pred_var),
                                         feed_dict={model.condition: False})

        pred_train = ds.denormalize(pred_train, 'out')[0, :, :]
        gt_train = ds.denormalize(ds.train_out[0:1, :predict_size, :], 'out')[0, :, :]
        var_train = ds.denormalize(np.sqrt(var_train), 'out', shift=False)[0, :, :]
        lower = [pred_train[i, 0] - 1.96 * var_train[i, 0] for i in range(predict_size)]
        upper = [pred_train[i, 0] + 1.96 * var_train[i, 0] for i in range(predict_size)]

        plt.figure(1, figsize=(6, 4))
        plt.plot(gt_train[:, 0], label='ground truth')
        plt.plot(pred_train[:, 0], label='prediction')
        plt.fill_between(range(predict_size), lower, upper,
                         color=(255. / 255., 178. / 255., 110. / 255.))
        plt.legend(loc=2)
        plt.grid(True)
        plt.xlabel("time (steps)")
        plt.xlim([0, predict_size])
        plt.savefig(self.out_dir + '/predict_train.pdf', bbox_inches='tight')
        plt.close(1)

        scipy.io.savemat(self.out_dir + '/predict_train.mat',
                         {'mean': pred_train, 'std': var_train, 'gt': gt_train})

        # Test
        model.load_ds(sess, ds.test_in[0:1, :predict_size, :],
                      ds.test_out[0:1, :predict_size, :])
        pred_test, var_test = sess.run((model.pred_mean, model.pred_var),
                                       feed_dict={model.condition: False})

        pred_test = ds.denormalize(pred_test, 'out')[0, :, :]
        gt_test = ds.denormalize(ds.test_out[0:1, :predict_size, :], 'out')[0, :, :]
        var_test = ds.denormalize(np.sqrt(var_test), 'out', shift=False)[0, :, :]
        lower = [pred_test[i, 0] - 1.96 * var_test[i, 0] for i in range(predict_size)]
        upper = [pred_test[i, 0] + 1.96 * var_test[i, 0] for i in range(predict_size)]

        plt.figure(1, figsize=(6, 4))
        plt.plot(gt_test[:, 0], label='ground truth')
        plt.plot(pred_test[:, 0], label='prediction')
        plt.fill_between(range(predict_size), lower, upper,
                         color=(255. / 255., 178. / 255., 110. / 255.))
        plt.legend(loc=2)
        plt.grid(True)
        plt.xlabel("time (steps)")
        plt.xlim([0, predict_size])
        plt.savefig(self.out_dir + '/predict_test.pdf', bbox_inches='tight')
        plt.close(1)

        scipy.io.savemat(self.out_dir + '/predict_test.mat',
                         {'mean': pred_test, 'std': var_test, 'gt': gt_test})

    def test_mse(self, sess):
        print("  test mse")
        model = self.model
        ds = self.ds

        mse_all = []
        for i in range(ds.test_in.shape[0]):
            model.load_ds(sess, ds.test_in[i:i+1, :, :], ds.test_out[i:i+1, :, :])
            pred = model.run(sess, model.pred_mean, {model.condition: False})[0]
            pred = ds.denormalize(pred, 'out')[0]
            gt = ds.denormalize(ds.test_out[i:i+1, :, :], 'out')[0]
            mse = sklearn.metrics.mean_squared_error(gt, pred)
            mse_all.append(mse)

        mse_all = np.mean(np.asarray(mse_all))
        rmse_all = math.sqrt(mse_all)
        text_file = open(self.out_dir + '/mse.txt', 'w')
        text_file.write("MSE:  %f\n" % mse_all)
        text_file.write("RMSE: %f\n" % rmse_all)
        text_file.close()
        self.last_rmse = rmse_all

    def var_dump(self, sess):
        print("  var dump")
        model = self.model

        text_file = open(self.out_dir + '/var_dump.txt', 'w')

        for name, variable in model.var_dict.items():
            value = sess.run(variable, feed_dict={model.condition: False})
            text_file.write(name + ":\n")

            if len(value.shape) == 1:
                for val in value:
                    text_file.write("  % .4e" % val)
            elif len(value.shape) == 2:
                for row in value:
                    for val in row:
                        text_file.write("  % .4e" % val)
                    text_file.write('\n')

            text_file.write("\n\n")

        text_file.close()
