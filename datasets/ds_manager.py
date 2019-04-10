import scipy.io
import numpy as np


class DSManager:
    """Conventions:
    * x[i+1] = f(x[i], u[i]), y[i] = g([x[i])
    * u, x, y have shape [ds_size, dimension]"""

    @staticmethod
    def load_ds(filename, normalize=False, print_title=True, dtype=np.float64):
        ds = scipy.io.loadmat(filename)
        if print_title:
            print('Loaded Dataset ' + ''.join(ds['title']))
        u = ds['ds_u'].astype(dtype)
        x = ds['ds_x'].astype(dtype)
        y = ds['ds_y'].astype(dtype)
        if normalize:
            u = DSManager.normalize_ds(u)
            x = DSManager.normalize_ds(x)
            y = DSManager.normalize_ds(y)
        return u, x, y

    @staticmethod
    def save_ds(filename, u, x, y, title, dtype=np.float64):
        assert(len(u.shape) == 2)
        assert(len(x.shape) == 2)
        assert(len(y.shape) == 2)
        assert(u.shape[0] == x.shape[0])  # last u was not used to propagate state
        assert(x.shape[0] == y.shape[0])
        scipy.io.savemat(filename, {'ds_u': u.astype(dtype),
                                    'ds_x': x.astype(dtype),
                                    'ds_y': y.astype(dtype),
                                    'title': title})

    @staticmethod
    def sample_ds_matrix(ds, ds_size, u_fn):
        """assumes the dataset uses matrices / column vectors as arguments"""
        u_all = []
        x_all = []
        y_all = []

        for i in range(ds_size):
            x = ds.get_state()
            x_all.append(np.asarray(x.T)[0, :])

            y = ds.measure()
            y_all.append(np.asarray(y.T)[0, :])

            u = u_fn(i, x)
            u_all.append(np.asarray(u.T)[0, :])
            ds.propagate(u)

        u_all = np.asarray(u_all)
        x_all = np.asarray(x_all)
        y_all = np.asarray(y_all)

        return u_all, x_all, y_all

    @staticmethod
    def sample_ds(ds, ds_size, u_fn):
        """assumes the dataset uses np arrays as arguments"""
        u_all = []
        x_all = []
        y_all = []

        for i in range(ds_size):
            x = ds.get_state()
            x_all.append(x)

            y_all.append(ds.measure())

            u = u_fn(i, x)
            u_all.append(u)
            ds.propagate(u)

        u_all = np.asarray(u_all)
        x_all = np.asarray(x_all)
        y_all = np.asarray(y_all)

        return u_all, x_all, y_all

    @staticmethod
    def normalize_ds(data):
        ret = data - np.mean(data, axis=0)
        return ret / np.std(ret, axis=0)
