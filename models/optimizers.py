import numpy as np
import theano
import theano.tensor as T

class Adagrad:
    """
    From Duchi, et al. 2011: http://dl.acm.org/citation.cfm?id=2021068
    """

    def __init__(self, params, lr=.01):
        self._lr = lr
        self._gsums = {}
        for param_name in params:
            param = params[param_name]
            self._gsums[param_name] = theano.shared(
                value=np.zeros_like(
                    param.get_value(borrow=True),
                    dtype=theano.config.floatX,
                ),
                name='gsum{}'.format(param_name),
            )

    def updates(self, cost, params):
        grads = [T.grad(cost, params[param_name]) for param_name in params]
        updates = []
        for grad, param_name in zip(grads, params):
            param = params[param_name]
            gsum_param = self._gsums[param_name]

            gsum_update = gsum_param + (grad ** 2)
            update = param - self._lr * (grad / T.sqrt(gsum_update + 1e-7))

            updates.append((gsum_param, gsum_update))
            updates.append((param, update))

        return updates

class Adadelta:

    """
    Based on the implementation from Keras:
    https://github.com/fchollet/keras

    From Zeiler, 2012: http://arxiv.org/abs/1212.5701
    """

    def __init__(self, params, lr=1.0, rho=.95, epsilon=1e-7):
        self._lr = lr
        self._rho = rho
        self._epsilon = epsilon
        self._accumulators = {}
        self._delta_accumulators = {}

        for param_name in params:
            param = params[param_name]
            self._accumulators[param_name] = theano.shared(
                value=np.zeros_like(
                    param.get_value(borrow=True),
                    dtype=theano.config.floatX,
                ),
                name='accum_{}'.format(param_name),
            )

            self._delta_accumulators[param_name] = theano.shared(
                value=np.zeros_like(
                    param.get_value(borrow=True),
                    dtype=theano.config.floatX,
                ),
                name='delta_accum_{}'.format(param_name)
            )

    def updates(self, cost, params):
        updates = []
        for param_name in params:
            param = params[param_name]
            accu = self._accumulators[param_name]
            delta_accu = self._delta_accumulators[param_name]

            grad = T.grad(cost, param)
            new_accu = self._rho * accu + (1 - self._rho) * (grad ** 2)
            update = grad * T.sqrt(delta_accu + self._epsilon) / T.sqrt(new_accu + self._epsilon)
            new_param = param - self._lr * update
            new_delta_accu = self._rho * delta_accu + (1 - self._rho) * (update ** 2)

            updates.append((accu, new_accu))
            updates.append((param, new_param))
            updates.append((delta_accu, new_delta_accu))

        return updates
