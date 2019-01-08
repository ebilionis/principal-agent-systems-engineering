"""
Transfer functions.

"""


__all__ = ['TransferFunction']


import theano
import theano.tensor as T
from _function import Function


class TransferFunction(Function):
    """
    An abstract class for transfer functions.

    :param t_q: The symbolic variable for quality.
    :param num_a:   The number of parameters (immutable).
    :param t_a: The symbolic variable for the parameters.
    :param t_t: The symbolic variable for the transfer function.
    """

    def __init__(self, t_q, num_a, t_a, t_t):
        self._num_a = num_a
        super(TransferFunction, self).__init__([t_q, t_a], t_t)

    @property
    def num_a(self):
        """
        Get the number of parameters.
        """
        return self._num_a

    @property
    def t_q(self):
        """
        Get the symbolic variable for the quality.
        """
        return self.t_x[0]

    @property
    def t_a(self):
        """
        Get the symbolic variable for the effort.
        """
        return self.t_x[1]

    @property
    def t_t(self):
        """
        Get the symbolic variable for the transfer function.
        """
        return self.t_t


class RequirementTransferFunction(TransferFunction):
    """
    A transfer function that pays a fixed participation amount and then
    a fixed amount if a requirement is met. It has three (3) parameters.

    :param t_q: The symbolic variable for quality (created if None)
    :param t_a: The symbolic variable for the parameters (created if None)
    :param gamma:   The step function is represented by a sigmoid. This
                    parameter gives the sharpness of the sigmoid. (immutable)
    """

    def __init__(self, t_q=None, t_a=None, gamma=10.):
        self._gamma = gamma
        if t_q is None:
            t_q = T.dscalar('q')
        if t_a is None:
            t_a = T.dvector('a')
        t_t = t_a[0] + t_a[1] / (1. + T.exp(gamma * (t_a[2] - t_q)))
        super(RequirementTransferFunction, self).__init__(t_q, 3, t_a, t_t)

    @property
    def gamma(self):
        """
        Get the gamma parameter.
        """
        return self._gamma


class RequirementPlusIncentiveTransferFunction(TransferFunction):
    """
    A transfer function that pays a fixed 
    """
    pass


if __name__ == '__main__':
    tr = RequirementTransferFunction()
    tr.compile()
    a = [0.05, 0.3, 1.]
    for q in [0.1, 0.6, 0.8, 1., 1.2]:
        print 'tr(%1.2f, [%1.2f, %1.2f, %1.2f]) = %1.2f' \
                % (q, a[0], a[1], a[2], tr(q, a))
