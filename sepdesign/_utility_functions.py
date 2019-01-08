"""
Utility functions.

"""



_all__ = ["UtilityFunction", "ExponentialUtilityFunction"]
import theano
import theano.tensor as T
from _function import Function


class UtilityFunction(Function):
    """
    An abstract class for the utility functions.

    :param t_pi:    A symbolic variable representing the payoff.
    :param t_util:  TODO Write
    """

    def __init__(self, t_pi, t_util):
        super(UtilityFunction, self).__init__([t_pi], t_util)

    @property
    def t_pi(self):
        return self.t_x[0]


    @property
    def t_util(self):
        return self.t_f



class ExponentialUtilityFunction(UtilityFunction):
    """
    This is the exponential utility function for different risk behaviors.
    """


class RiskAverseUtilityFunction(UtilityFunction):
    """
    The utiltiy function for risk averse behavior
    """

    def __init__(self, t_pi=None, risk_intensity = -2.0):
        if t_pi == None:
            t_pi = T.dscalar('pi')
        self.a = 1.0 / (1.0 - T.exp(risk_intensity))
        self.b = self.a
        t_util = self.a - self.b * T.exp(risk_intensity * t_pi)
        super(RiskAverseUtilityFunction, self).__init__(t_pi, t_util)


class RiskNeutralUtilityFunction(UtilityFunction):
    """
    The utiltiy function for risk neutral behavior
    """

    def __init__(self, t_pi=None):
        if t_pi == None:
            t_pi = T.dscalar('pi')
        t_util = t_pi
        super(RiskNeutralUtilityFunction, self).__init__(t_pi, t_util)


class RiskProneUtilityFunction(UtilityFunction):
    """
    The utiltiy function for risk prone behavior
    """

    def __init__(self, t_pi=None, risk_intensity = 2):

        if t_pi == None:
            t_pi = T.dscalar('pi')

        self.a = 1.0 / (1.0 - T.exp(risk_intensity))
        self.b = self.a
        t_util = self.a - self.b * T.exp(risk_intensity * t_pi)
        super(RiskProneUtilityFunction, self).__init__(t_pi, t_util)


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    pi = np.linspace(0,1, 100)
    u_averse  = RiskAverseUtilityFunction(risk_intensity=-3)
    u_neutral = RiskNeutralUtilityFunction()
    u_prone   = RiskProneUtilityFunction(risk_intensity=3)

    u_averse.compile()
    u_neutral.compile()
    u_prone.compile()

    ua = []
    un = []
    up = []
    for i in pi:
        ua += [u_averse(i)]
        un += [u_neutral(i)]
        up += [u_prone(i)]

    plt.plot(pi, ua, label='risk averse')
    plt.plot(pi, un, label='risk neutral')
    plt.plot(pi, up, label='risk prone')

    plt.xlabel(r'$\pi$')
    plt.ylabel(r'utility')

    plt.legend()
    plt.savefig('utility.png', dpi = 300)




















