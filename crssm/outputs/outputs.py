import os
import math
import sklearn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.special
import scipy.io


# TODO: clean up, s.t. outputs clearer and more useful

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
        self.hidden_states(sess)
        self.test_mse(sess)
        self.variance_mse(sess)
        self.mse_norecog(sess)
        self.particles(sess)
        self.var_dump(sess)
        self.variance_vs_time(sess)

    def training_stats(self):
        if self.trainer is not None:
            print("  training stats")
            plt.figure(1)
            plt.plot(self.trainer.train_all, label='train')
            plt.plot(self.trainer.test_all, label='test')
            plt.legend()
            plt.title('Negative Log-Likelihood')
            plt.savefig(self.out_dir + '/training.eps')
            plt.close(1)

    def variance_mse(self, sess):
        model = self.model
        ds = self.ds
        recog_len = model.config['recog_len']
        mse_summary = []
        for pred_len in [1, 10, 100, 1000]:

            # Create Batches
            batch_len = pred_len + recog_len
            if batch_len > ds.train_in.shape[1]:
                break
            batch_stride = int(batch_len/2) + 1
            print("  variance seq %d" % pred_len)
            x_train_in_batch, x_train_out_batch, x_test_in_batch, x_test_out_batch =\
                ds.get_batches(batch_len, batch_stride)

            # Train
            model.load_ds(sess, x_train_in_batch, x_train_out_batch)
            res = model.run(sess, (model.mse, model.sde),
                            {model.condition: False})
            mse_all_train = float(np.mean(res[0]))
            sde_all_train = res[1]
            sde_all_train = sde_all_train[:, recog_len:, :]

            # Test
            model.load_ds(sess, x_test_in_batch, x_test_out_batch)
            res = model.run(sess, (model.mse, model.sde),
                            {model.condition: False})
            mse_all_test = float(np.mean(res[0]))
            sde_all_test = res[1]
            sde_all_test = sde_all_test[:, recog_len:, :]

            # Plot
            cnt_train = sde_all_train.shape[0] * sde_all_train.shape[1] * ds.dim_y
            cnt_test = sde_all_test.shape[0] * sde_all_test.shape[1] * ds.dim_y
            var_max = 3

            plt.figure(1, figsize=(3.7, 3))
            x_axis = np.linspace(0, var_max, 1000)
            y_axis = np.asarray([(sde_all_train < x).sum()/cnt_train for x in x_axis])
            plt.plot(x_axis, y_axis, label='train')
            y_axis = np.asarray([(sde_all_test < x).sum()/cnt_test for x in x_axis])
            plt.plot(x_axis, y_axis, label='test')
            error_y = scipy.special.erf(x_axis / math.sqrt(2))
            plt.plot(x_axis, error_y, 'k--', label='ref')
            plt.legend(loc=4)
            plt.xlabel('threshold (SD)')
            plt.ylabel('inliers (ratio)')
            plt.xlim([0, var_max])
            plt.ylim([0, 1])
            plt.grid(True)
            plt.savefig(self.out_dir + '/variance_norecog_%d.eps' % pred_len, bbox_inches='tight')
            plt.close(1)

            mse_summary.append((batch_len, mse_all_train, mse_all_test))

        # MSE
        print("  batch mse")
        text_file = open(self.out_dir + '/mse_normalized.txt', 'w')
        for l, train, test in mse_summary:
            text_file.write("%04d Train MSE: %f\n" % (l, train))
            text_file.write("%04d Test MSE:  %f\n\n" % (l, test))
        text_file.close()

    def mse_norecog(self, sess):
        print("  mse norecog")
        model = self.model
        ds = self.ds
        err_begin = 16

        # Predictions
        seq_len = 150
        if ds.train_in.shape[1] < seq_len:
            seq_len = ds.train_in.shape[1]
        x_train_in_batch, x_train_out_batch, x_test_in_batch, x_test_out_batch = ds.get_batches(seq_len, 1)
        model.load_ds(sess, x_test_in_batch, x_test_out_batch)
        data_should, data_is, sde_all = model.run(sess, (model.sample_out, model.pred_mean, model.sde),
                                                  {model.condition: False})
        data_should = ds.denormalize(data_should, 'out')
        data_is = ds.denormalize(data_is, 'out')
        mse_all = np.square(data_should-data_is)

        # MSE
        text_file = open(self.out_dir + '/mse_norecog.txt', 'w')
        for err_end in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            mse = np.mean(mse_all[:, err_begin:(err_begin + err_end), :])
            text_file.write("%f\n" % math.sqrt(mse))
        text_file.close()

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
        plt.savefig(self.out_dir + '/predict_train.eps', bbox_inches='tight')
        plt.close(1)

        scipy.io.savemat(self.out_dir + '/prediction_train.mat',
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
        plt.savefig(self.out_dir + '/predict_test.eps', bbox_inches='tight')
        plt.close(1)

        scipy.io.savemat(self.out_dir + '/prediction_test.mat',
                         {'mean': pred_test, 'std': var_test, 'gt': gt_test})

    def hidden_states(self, sess):
        print("  hidden states")
        model = self.model
        ds = self.ds

        # Train
        model.load_ds(sess, ds.train_in[0:1, :, :],
                      ds.train_out[0:1, :, :])
        pred_train, var_train = sess.run((model.internal_mean, model.internal_var),
                                         feed_dict={model.condition: False})
        plt.figure(1, figsize=(20, 20))
        plt.subplot(211)
        for i in range(pred_train.shape[2]):
            plt.plot(pred_train[0, :, i], label='x_%d' % i)
        for i in range(ds.train_in.shape[2]):
            plt.plot(ds.train_in[0, :, i], label='u_%d' % i)
        for i in range(ds.train_out.shape[2]):
            plt.plot(ds.train_out[0, :, i], label='y_%d' % i)
        plt.legend()
        plt.title('Train-Set')

        # Test
        plt.subplot(212)
        model.load_ds(sess, ds.test_in[0:1, :, :],
                      ds.test_out[0:1, :, :])
        pred_test, var_test = sess.run((model.internal_mean, model.internal_var),
                                       feed_dict={model.condition: False})
        plt.figure(1, figsize=(20, 8))
        for i in range(pred_test.shape[2]):
            plt.plot(pred_test[0, :, i], label='x_%d' % i)
        for i in range(ds.test_in.shape[2]):
            plt.plot(ds.test_in[0, :, i], label='u_%d' % i)
        for i in range(ds.test_out.shape[2]):
            plt.plot(ds.test_out[0, :, i], label='y_%d' % i)
        plt.legend()
        plt.title('Test-Set')
        plt.savefig(self.out_dir + '/internal.eps')
        plt.close(1)

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
        text_file = open(self.out_dir + '/mse_denormalized.txt', 'w')
        text_file.write("MSE:  %f\n" % mse_all)
        text_file.write("RMSE: %f\n" % rmse_all)
        text_file.close()
        self.last_rmse = rmse_all

    def particles(self, sess, predict_size=512):
        print("  multistep prediction")
        model = self.model
        ds = self.ds

        model.load_ds(sess, ds.test_in[0:1, :predict_size, :],
                      ds.test_out[0:1, :predict_size, :])
        particles = sess.run(model.y_final,
                             feed_dict={model.condition: False})

        scipy.io.savemat(self.out_dir + '/particles.mat',
                         {'p': particles[0]})

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

    def variance_vs_time(self, sess, recog_size=50, predict_size=350):
        print("  variance vs time")
        model = self.model
        ds = self.ds

        model.load_ds(sess, ds.test_in[0:1, :predict_size, :],
                      ds.test_out[0:1, :predict_size, :])
        var_test = sess.run(model.pred_var,
                            feed_dict={model.condition: False})
        var_test = var_test[0, recog_size:, :]

        plt.figure(1, figsize=(13.968*0.5/2.54, 4.5/2.54))
        plt.plot(np.mean(var_test, axis=1))
        plt.grid(True)
        plt.xlabel("time (steps)")
        plt.ylabel("variance")
        plt.xlim([0, predict_size-recog_size])
        plt.ylim([0, 0.16])
        plt.tight_layout(pad=0.2)
        plt.savefig(self.out_dir + '/variance_vs_time.pdf')
        plt.close(1)
