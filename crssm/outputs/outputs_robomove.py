import matplotlib.pyplot as plt
from crssm.outputs.outputs import Outputs


class OutputsRoboMove(Outputs):

    def __init__(self, *args):
        super(OutputsRoboMove, self).__init__(*args)

    def _create_all(self, sess):
        super(OutputsRoboMove, self)._create_all(sess)
        self.robomove_prediction(sess)

    def robomove_prediction(self, sess, predict_size=300):
        print("  robomove prediction")
        model = self.model
        ds = self.ds

        # Train
        model.load_ds(sess, ds.train_in[0:1, :predict_size, :],
                      ds.train_out[0:1, :predict_size, :])
        pred_train, var_train = sess.run((model.pred_mean, model.pred_var),
                                         feed_dict={model.condition: False})
        pred_train = pred_train[0, :, :]

        plt.figure(1, figsize=(6, 5))
        plt.plot(ds.train_out[0, :predict_size, 0], ds.train_out[0, :predict_size, 1], '*-', label='ground truth')
        plt.plot(pred_train[:, 0], pred_train[:, 1], '*-', label='prediction')
        plt.legend(loc=2)
        plt.axis('equal')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(self.out_dir + '/robomove_train.pdf', bbox_inches='tight')
        plt.close(1)

        # Test
        model.load_ds(sess, ds.test_in[0:1, :predict_size, :],
                      ds.test_out[0:1, :predict_size, :])
        pred_test, var_test = sess.run((model.pred_mean, model.pred_var),
                                       feed_dict={model.condition: False})
        pred_test = pred_test[0, :, :]

        plt.figure(1, figsize=(6, 5))
        plt.plot(ds.test_out[0, :predict_size, 0], ds.test_out[0, :predict_size, 1], '*-', label='ground truth')
        plt.plot(pred_test[:, 0], pred_test[:, 1], '*-', label='prediction')
        plt.legend(loc=2)
        plt.axis('equal')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(self.out_dir + '/robomove_test.pdf', bbox_inches='tight')
        plt.close(1)
