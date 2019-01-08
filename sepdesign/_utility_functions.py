"""
Utility functions

"""

import theano
import theano.tensor as T
from _function import Function


class UtilityFunction(Function):
    """
    An abstract class for the utility functions
    """

    def __init__(self, t_pi, t_util):
        super(UtilityFunction, self).__init__([t_pi], t_util)

    @property
    def t_pi(self):
        return self.t_x[0]


    @property
    def t_util(self):
        return self.t_f


class RiskAverseUtilityFunction(UtilityFunction):
    """
    The utiltiy function for risk averse behavior
    """

    def __init__(self, t_pi=None):
        if t_pi == None:
            t_pi = T.dvector('pi')
        t_util = T.flatten(t_pi)
        self.a = 1.0 / (1.0 - T.exp(-2))
        self.b = self.a
        t_util = T.flatten(self.a - self.b * T.exp(-2 * t_pi))
        super(RiskAverseUtilityFunction, self).__init__(t_pi, t_util)


class RiskNeutralUtilityFunction(UtilityFunction):
    """
    The utiltiy function for risk neutral behavior
    """

    def __init__(self, t_pi=None):

        if t_pi == None:
            t_pi = T.dvector('pi')
        t_util = T.flatten(t_pi)
        super(RiskNeutralUtilityFunction, self).__init__(t_pi, t_util)


class RiskProneUtilityFunction(UtilityFunction):
    """
    The utiltiy function for risk prone behavior
    """

    def __init__(self, t_pi=None):

        if t_pi == None:
            t_pi = T.dvector('pi')

        self.a = 1.0 / (1.0 - T.exp(2))
        self.b = self.a
        t_util = T.flatten(self.a - self.b * T.exp(2 * t_pi))
        super(RiskProneUtilityFunction, self).__init__(t_pi, t_util)


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    pi = np.linspace(0,1, 100)
    u_averse  = RiskAverseUtilityFunction()
    u_neutral = RiskNeutralUtilityFunction()
    u_prone   = RiskProneUtilityFunction()

    u_averse.compile()
    u_neutral.compile()
    u_prone.compile()

    plt.plot(pi, u_averse(pi), label='risk averse')
    plt.plot(pi, u_neutral(pi), label='risk neutral')
    plt.plot(pi, u_prone(pi), label='risk prone')
    plt.xlabel(r'$\pi$')
    plt.ylabel(r'utility')

    plt.legend()
    plt.savefig('utility.png', dpi = 300)




















