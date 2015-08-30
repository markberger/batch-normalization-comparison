import numpy as np
import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams

from model import Model
from optimizers import Adagrad

DEFAULT_DROPOUT = 0.5

class MLP(Model):

    def __init__(self, layers, dropout_p=DEFAULT_DROPOUT):
        super(MLP, self).__init__()

        self._Xs = theano.shared(
            value=np.zeros((1, 1), dtype=theano.config.floatX),
            name='Xs')
        self._Ys = theano.shared(
            value=np.zeros((1, 1), dtype=theano.config.floatX),
            name='Ys')
        self._dropout_p = dropout_p
        self._h_activation = T.nnet.sigmoid

        self._params = {}
        self._layers = []
        for i, layer in enumerate(layers):
            w_label = 'W_{}'.format(i)
            w_bound = np.sqrt(layer[0] + layer[1])
            self._params[w_label] = theano.shared(
                value=np.random.uniform(
                    -1.0/w_bound,
                    1.0/w_bound,
                    layer).astype(theano.config.floatX),
                name=w_label
            )

            b_label = 'b_{}'.format(i)
            self._params[b_label] = theano.shared(
                value=np.zeros((layer[1],)).astype(theano.config.floatX),
                name=b_label,
            )

            self._layers.append((w_label, b_label))

        self._compile()

    def _compile(self):
        Xs = T.matrix('Xs')
        Ys = T.matrix('Ys')
        index = T.lscalar('index')
        batch_size = T.lscalar('batch_size')

        Y_hats_training = self._feed_forward(Xs, dropout=True)
        training_cost = self._cost(Ys, Y_hats_training)

        Y_hats_out = self._feed_forward(Xs, dropout=False)
        testing_cost = self._cost(Ys, Y_hats_out)

        self._optimizer = Adagrad(self._params)
        updates = self._optimizer.updates(training_cost, self._params)

        self._batch_train = theano.function(
            inputs=[index, batch_size],
            outputs=training_cost,
            givens={
                Xs: self._Xs[index*batch_size:(index+1)*batch_size, :],
                Ys: self._Ys[index*batch_size:(index+1)*batch_size, :],
            },
            updates=updates
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

    def _feed_forward(self, Xs, dropout=False):
        srng = RandomStreams()
        curr = Xs

        for w_label, b_label in self._layers:
            W = self._params[w_label]
            b = self._params[b_label]
            curr = self._h_activation(curr.dot(W) + b)

            if dropout:
                curr *= srng.binomial(size=curr.shape, p=self._dropout_p)
            else:
                curr *= self._dropout_p

        return T.nnet.softmax(curr)

    def _cost(self, Ys, Y_hats):
        return T.nnet.categorical_crossentropy(Y_hats, Ys).mean()
