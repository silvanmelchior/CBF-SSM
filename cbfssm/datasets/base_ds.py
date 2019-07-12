import os
import numpy as np


class BaseDS:

    dim_u = None
    dim_y = None

    def __init__(self, seq_len, seq_stride):
        self.seq_len = seq_len
        self.seq_stride = seq_stride
        self.train_in = np.empty(0)
        self.train_out = np.empty(0)
        self.test_in = np.empty(0)
        self.test_out = np.empty(0)
        self.train_in_batch = np.empty(0)
        self.train_out_batch = np.empty(0)
        self.test_in_batch = np.empty(0)
        self.test_out_batch = np.empty(0)
        self.mean = {'in': np.empty(()), 'out': np.empty(())}
        self.std = {'in': np.empty(()), 'out': np.empty(())}
        self.data_path = os.path.dirname(__file__) + '/data/'

    def normalize_init(self, data_in, data_out):
        assert len(data_in.shape) == 2
        assert len(data_out.shape) == 2
        self.mean['in'] = np.mean(data_in, axis=0)
        self.std['in'] = np.std(data_in - self.mean['in'], axis=0)
        self.mean['out'] = np.mean(data_out, axis=0)
        self.std['out'] = np.std(data_out - self.mean['out'], axis=0)

    def normalize(self, data, key):
        return (data - self.mean[key]) / self.std[key]

    def denormalize(self, data, key, shift=True):
        res = data * self.std[key]
        if shift:
            return res + self.mean[key]
        else:
            return res

    def get_batches(self, seq_len, seq_stride):
        return (self.rnn_batches(self.train_in, seq_len, seq_stride, 0),
                self.rnn_batches(self.train_out, seq_len, seq_stride, 0),
                self.rnn_batches(self.test_in, seq_len, seq_stride, 0),
                self.rnn_batches(self.test_out, seq_len, seq_stride, 0))

    def create_batches(self):
        self.train_in_batch, self.train_out_batch, self.test_in_batch, self.test_out_batch\
            = self.get_batches(self.seq_len, self.seq_stride)
        self.print_stats()

    @staticmethod
    def rnn_batches(x, length, stride, _):
        """Split the data x into batches of length every stride.

        assumes x has shape [experiments, time-samples, dimension).
        """
        assert x.ndim == 3, "data must be shaped as [experiments x time x dimension]"

        def rnn_batches_ex(x_):
            num_points, dim = x_.shape
            assert num_points >= length, "Sequence length must be shorter than data."

            data_batch = [x_[i:i + length, :]
                          for i in range(0, num_points - length + 1, stride)]

            # Make sure last data points are contained in a batch
            remainder = (num_points - length) % stride
            if remainder > 0:
                data_batch.append(x_[-length:, :])

            return np.stack(data_batch, axis=0)

        batches = [rnn_batches_ex(ex) for ex in x]
        return np.concatenate(batches, axis=0)

    def print_stats(self):
        print('Dataset Stats:')
        print('  sequence length: %d' % self.seq_len)
        print('  train samples: %d' % (self.train_in.shape[0]*self.train_in.shape[1]))
        print('  train sequences: %d' % self.train_in_batch.shape[0])
        print('  test samples: %d' % (self.test_in.shape[0]*self.test_in.shape[1]))
        print('  test sequences: %d' % self.test_in_batch.shape[0])
