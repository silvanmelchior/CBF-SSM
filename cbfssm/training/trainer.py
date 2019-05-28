import numpy as np
import tensorflow as tf

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x


class Trainer:

    def __init__(self, model, model_dir):
        self.model = model
        self.model_dir = model_dir
        self.train_all = []
        self.test_all = []

    def train(self, ds, epochs, retrain=False):
        print('\nTraining...\n')
        model = self.model
        with model.graph.as_default():
            config = tf.ConfigProto()
            # Pararellizing ops (default=2)
            config.intra_op_parallelism_threads = 5
            # Executing ops in parallel (default=5)
            config.inter_op_parallelism_threads = 10

            with tf.Session(config=config) as sess:

                if retrain:
                    model.saver.restore(sess, self.model_dir + '/model.ckpt')
                else:
                    sess.run(model.init)

                lowest_train = float('inf')
                for epoch in tqdm(range(epochs)):

                    # Train
                    model.load_ds(sess, ds.train_in_batch, ds.train_out_batch)
                    train_loss = model.run(sess, (model.train, model.loss),
                                           {model.condition: True})
                    train_loss = np.mean(train_loss[1])

                    # Test
                    model.load_ds(sess, ds.test_in_batch, ds.test_out_batch)
                    test_loss = model.run(sess, model.loss,
                                          {model.condition: True})
                    test_loss = np.mean(test_loss)

                    # Output
                    print('[{epoch:04}]: Train {train}, Test {test}'.format(
                        epoch=epoch, train=train_loss, test=test_loss))

                    self.train_all.append(train_loss)
                    self.test_all.append(test_loss)

                    # Save Best
                    if train_loss < lowest_train:
                        model.saver.save(sess, self.model_dir + '/best.ckpt')
                        lowest_train = train_loss

                # Save Last
                model.saver.save(sess, self.model_dir + '/model.ckpt')
