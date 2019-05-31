import sys
import numpy as np
import tensorflow as tf


class BaseModel:

    def __init__(self, config, dtype=tf.float32):
        self.config = config
        self.dtype = dtype
        self.graph = tf.Graph()
        self._build_ds_pipeline()
        self._build_graph()

    def _build_ds_pipeline(self):
        dim_u = self.config['ds'].dim_u
        dim_y = self.config['ds'].dim_y

        with self.graph.as_default():
            self.data_in = tf.placeholder(self.dtype, shape=[None, None, dim_u])  # [ds_size, seq_len, dim]
            self.data_out = tf.placeholder(self.dtype, shape=[None, None, dim_y])  # [ds_size, seq_len, dim]
            self.repeats = tf.placeholder(tf.int64)
            self.condition = tf.placeholder(tf.bool)
            dataset = tf.data.Dataset.from_tensor_slices((self.data_in, self.data_out))
            dataset = dataset.repeat(self.repeats).shuffle(self.config['shuffle'])
            dataset = dataset.batch(self.config['batch_size'])
            dataset = dataset.prefetch(buffer_size=1)
            self.dataset_iterator = dataset.make_initializable_iterator()
            self.sample_in, self.sample_out = self.dataset_iterator.get_next()  # [batch_size, seq_len, dim]
            self.batch_tf = tf.shape(self.sample_in)[0]  # size of current batch
            self.seq_len_tf = tf.shape(self.sample_in)[1]  # seq_len of current batch

    def _build_graph(self):
        pass

    def load_ds(self, sess, data_in, data_out, repeats=1):
        sess.run(self.dataset_iterator.initializer,
                 feed_dict={self.data_in: data_in,
                            self.data_out: data_out,
                            self.repeats: repeats})

    @staticmethod
    def run(sess, tensors, feed_dict, show_progress=False):
        res_all = None
        while True:
            try:
                res = sess.run(tensors, feed_dict=feed_dict)
                if show_progress:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                if not isinstance(res, tuple):
                    res = (res,)
                if res_all is None:
                    res_all = [i for i in res]
                    for i in range(len(res)):
                        if res_all[i] is not None:
                            res_all[i] = np.atleast_1d(res_all[i])
                else:
                    for i, item in enumerate(res):
                        if item is not None:
                            item = np.atleast_1d(item)
                            res_all[i] = np.concatenate(
                                (res_all[i], item), axis=0)
            except tf.errors.OutOfRangeError:
                break

        if show_progress:
            print()
        return res_all
