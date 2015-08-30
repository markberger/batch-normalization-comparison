import numpy as np
import theano

class Model(object):

    def _batch_predict(self, idx, batch_size):
        raise Exception('Not yet implemented')

    def _batch_train(self, idx, batch_size):
        raise Exception('Not yet implemented')

    def _batch_cost(self, y_true, y_hats):
        raise Exception('Not yet implemented')

    def batch_predict(self, Xs, batch_size):
        self._Xs.set_value(Xs)
        num_batches = int(np.ceil(float(Xs.shape[0]) / batch_size))
        outputs = []
        for i in xrange(num_batches):
            outputs.append(self._batch_predict(i, batch_size))
        return np.vstack(outputs)

    def batch_train(self, Xs, Ys, batch_size):
        self._Xs.set_value(Xs)
        self._Ys.set_value(Ys)
        num_batches = int(np.ceil(float(Xs.shape[0]) / batch_size))
        for i in xrange(num_batches):
            self._batch_train(i, batch_size)

    def batch_cost(self, Y_hats, Y_true, batch_size):
        num_batches = int(np.ceil(float(Y_hats.shape[0]) / batch_size))
        cost = 0
        for i in xrange(num_batches):
            y_hats = Y_hats[i*batch_size:(i+1)*batch_size,:]
            y_true = Y_true[i*batch_size:(i+1)*batch_size,:]
            cost += self._batch_cost(y_true, y_hats)
        return cost / num_batches

    def get_params(self):
        return self._params
