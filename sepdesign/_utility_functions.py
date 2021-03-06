"""
Utility functions.

"""


_all__ = ["UtilityFunction", "ExponentialUtilityFunction"]
import theano
import theano.tensor as T
from sepdesign._function import Function


class UtilityFunction(Function):
    """
    An abstract class for the utility functions.

    :param t_pi:    A symbolic variable representing the payoff.
    :param t_util:  TODO Write
    """

    def __init__(self, t_pi, t_util, name='UtilityFunction'):
        super(UtilityFunction, self).__init__([t_pi], t_util, name=name)

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

    def __init__(self, risk_preference=0.0, t_pi=None,
                 name='ExponentialUtilityFunction'):
        if t_pi == None:
            t_pi = T.dvector('pi')
        if risk_preference != 0.0:
            a = 1.0 / (1.0 - T.exp(risk_preference))
            b = a
            t_util = a - b * T.exp(risk_preference * t_pi)
        else:
            t_util = t_pi
        self._risk_preference = risk_preference
        super(ExponentialUtilityFunction, self).__init__(t_pi, t_util,
                                                         name=name)

    @property
    def risk_preference(self):
        """
        Get the risk preference.
        """
        return self._risk_preference

    def __repr__(self):
        """
        Get a string representation of the object.
        """
        return super(ExponentialUtilityFunction, self).__repr__() + \
                '(risk_preference=%1.2f)' % self.risk_preference


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    pis = np.linspace(0,1, 100)
    u_averse  = ExponentialUtilityFunction(risk_preference = -2.0)
    u_neutral = ExponentialUtilityFunction(risk_preference =  0.0)
    u_prone   = ExponentialUtilityFunction(risk_preference =  2.0)
    print(str(u_averse))
    print(str(u_neutral))
    print(str(u_prone))

    u_averse.compile()
    u_neutral.compile()
    u_prone.compile()

    ua = u_averse(pis)
    un = u_neutral(pis)
    up = u_prone(pis)

    plt.plot(pis, ua, label='risk averse')
    plt.plot(pis, un, label='risk neutral')
    plt.plot(pis, up, label='risk prone')

    plt.xlabel(r'$\pi$')
    plt.ylabel(r'utility')

    plt.legend()
    plt.show()

