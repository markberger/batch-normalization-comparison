import numpy as np
import theano
import theano.tensor as T

from model import Model
from optimizers import Adagrad

DEFAULT_EPSILON = 1e-6
DEFAULT_RHO = 0.9

class BNN(Model):

    def __init__(self, layers, rho=DEFAULT_RHO, epsilon=DEFAULT_EPSILON):
        super(BNN, self).__init__()

        self._Xs = theano.shared(
            value=np.zeros((1, 1), dtype=theano.config.floatX),
            name='Xs')
        self._Ys = theano.shared(
            value=np.zeros((1, 1), dtype=theano.config.floatX),
            name='Ys')
        self._h_activation = T.nnet.sigmoid
        self._rho = rho
        self._epsilon = epsilon

        self._params = {}
        self._running_avgs = {}
        self._layers = []
        for i, layer in enumerate(layers):
            w_label = 'W_{}'.format(i)
            w_bound = np.sqrt(layer[0] + layer[1])
            self._params[w_label] = theano.shared(
                value=np.random.uniform(
                    -1.0/w_bound,
                    1.0/w_bound,
                    layer).astype(theano.config.floatX),
                name=w_label,
            )

            b_label = 'b_{}'.format(i)
            self._params[b_label] = theano.shared(
                value=np.zeros((layer[1],)).astype(theano.config.floatX),
                name=b_label,
            )

            mean_label = 'mean_{}'.format(i)
            self._running_avgs[mean_label] = theano.shared(
                value=np.zeros((layer[0],)),
                name=mean_label,
            )

            std_label = 'std_{}'.format(i)
            self._running_avgs[std_label] = theano.shared(
                value=np.ones((layer[0],))
            )

            self._layers.append((w_label, b_label, mean_label, std_label))

        self._compile()

    def _compile(self):
        Xs = T.matrix('Xs')
        Ys = T.matrix('Ys')
        index = T.lscalar('index')
        batch_size = T.lscalar('batch_size')

        Y_hats_training, average_updates = self._feed_forward(Xs, training=True)
        training_cost = self._cost(Ys, Y_hats_training)

        Y_hats_out, _ = self._feed_forward(Xs, training=False)
        testing_cost = self._cost(Ys, Y_hats_out)

        self._optimizer = Adagrad(self._params)
        updates = self._optimizer.updates(training_cost, self._params)
        all_updates = updates + average_updates

        self._batch_train = theano.function(
            inputs=[index, batch_size],
            outputs=training_cost,
            givens={
                Xs: self._Xs[index*batch_size:(index+1)*batch_size, :],
                Ys: self._Ys[index*batch_size:(index+1)*batch_size, :],
            },
            updates=all_updates
        )

        self._batch_predict = theano.function(
            inputs=[index, batch_size],
            outputs=Y_hats_out,
            givens={
                Xs: self._Xs[index*batch_size:(index+1)*batch_size, :],
            },
        )

        self._batch_cost = theano.function(
            inputs=[Ys, Y_hats_out],
            outputs=testing_cost,
        )

    def _feed_forward(self, Xs, training=False):
        updates = []
        curr = Xs

        for w_label, b_label, mean_label, std_label in self._layers:
            W = self._params[w_label]
            b = self._params[b_label]
            avg_mean = self._running_avgs[mean_label]
            avg_std = self._running_avgs[std_label]

            curr_mean = curr.mean(axis=0)
            curr_std = T.mean((curr - curr_mean) ** 2, axis=0)

            mean_update = self._rho * avg_mean + (1 - self._rho) * curr_mean
            updates.append((avg_mean, mean_update))

            std_update = self._rho * avg_std + (1 - self._rho) * curr_std
            updates.append((avg_std, std_update))

            if training:
                curr_x = (curr - curr_mean) / ((curr_std + self._epsilon) ** 0.5)
                curr = self._h_activation(curr_x.dot(W) + b)
            else:
                W_norm = W / ((avg_std.dimshuffle(0,'x') + self._epsilon) ** 0.5)
                b_norm = b - avg_mean.dot(W_norm)
                curr = self._h_activation(curr.dot(W_norm) + b_norm)

        return T.nnet.softmax(curr), updates

    def _cost(self, Ys, Y_hats):
        return T.nnet.categorical_crossentropy(Y_hats, Ys).mean()
