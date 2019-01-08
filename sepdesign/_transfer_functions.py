"""
Transfer functions.

"""


__all__ = ['TransferFunction', 'RequirementTransferFunction',
           'RequirementPlusIncentiveTransferFunction']


import theano
import theano.tensor as T
import numpy as np
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
        return self.t_f

    def plot(self, ax, a, qs=None, **kwargs):
        """
        Plot the transfer function in ax using the parameters and the
        qualities qs.
        """
        if qs is None:
            qs = np.linspace(0, 2, 100)
        ts = np.array([self(q, a) for q in qs])
        ax.plot(qs, ts, **kwargs)


class RequirementTransferFunction(TransferFunction):
    """
    A transfer function that pays a fixed participation amount and then
    a fixed amount if a requirement is met. It has three (3) parameters.

    :param t_q: The symbolic variable for quality (created if None)
    :param t_a: The symbolic variable for the parameters (created if None)
    :param gamma:   The step function is represented by a sigmoid. This
                    parameter gives the sharpness of the sigmoid. (immutable)
    """

    def __init__(self, t_q=None, t_a=None, gamma=50.):
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
    A transfer function that pays a fixed participation amount, a fixed
    amount if a requirement is met, and then linearly after the requirement
    is met. It has four (4) parameters.

    :param t_q: The symbolic variable for quality (created if None)
    :param t_a: The symbolic variable for the parameters (created if None)
    :param gamma:   The step function is represented by a sigmoid. This
                    parameter gives the sharpness of the sigmoid. (immutable)
    """
    
    def __init__(self, t_q=None, t_a=None, gamma=50.):
        self._gamma = gamma
        if t_q is None:
            t_q = T.dscalar('q')
        if t_a is None:
            t_a = T.dvector('a')
        t_t = t_a[0] \
                + (t_a[1] + t_a[3] * (t_q - t_a[2])) \
                / (1. + T.exp(gamma * (t_a[2] - t_q)))
        super(RequirementPlusIncentiveTransferFunction, 
                self).__init__(t_q, 4, t_a, t_t)

    @property
    def gamma(self):
        """
        Get the gamma parameter.
        """
        return self._gamma


if __name__ == '__main__':
    tr = RequirementTransferFunction()
    tr.compile()
    a1 = [0.05, 0.3, 1.]
    for q in [0.1, 0.6, 0.8, 1., 1.2]:
        print 'tr(%1.2f, [%1.2f, %1.2f, %1.2f]) = %1.2f' \
                % (q, a1[0], a1[1], a1[2], tr(q, a1))

    trpi = RequirementPlusIncentiveTransferFunction()
    trpi.compile()
    a2 = [0.05, 0.3, 1., 0.6]
    for q in [0.1, 0.6, 0.8, 1., 1.2]:
        print 'tr(%1.2f, [%1.2f, %1.2f, %1.2f, %1.2f]) = %1.2f' \
                % (q, a2[0], a2[1], a2[2], a2[3], trpi(q, a2))

    # Let's plot them
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('white')
    sns.set_context('paper')
    fig, ax = plt.subplots()
    tr.plot(ax, a1, lw=2, label='R')
    trpi.plot(ax, a2, lw=2, label='RPI')
    plt.show()
