"""
Class of value functions
"""

__all__ = ["ValueFunction", "RequirementValueFunction"]


import theano
import theano.tensor as T
import numpy as np
from _function import Function


class ValueFunction(Function):
    """
    This is an abstract class for value functions.
    """
    def __init__(self, t_q, t_req, num_a, t_a, t_v):
        """
        t_q:  The vector pf qaulities
        t_re: The min9imum requirement for the system to have nonzero value
        t_a:  The coefficient that needed
        """
        self._t_req = t_req
        self._num_a = num_a
        if self._num_a != 0:
            super(ValueFunction, self).__init__([t_q, t_req, t_a], t_v)
        else:
            super(ValueFunction, self).__init__([t_q, t_req], t_v)

    @property
    def t_req(self):
        """
        Get the value of the t_req.
        """
        return self._t_req


class RequirementValueFunction(ValueFunction):
    """
    This is a 0 or 1 value function
    """
    def __init__(self, t_q=None, t_req=None, t_a = None, gamma = 50.0):
        self._gamma = gamma
        if t_q == None:
            t_q = T.dvector('q')
        if t_req == None:
            t_req = T.dscalar('req')
        t_v = 1. / (1. + T.exp(gamma * (t_req - t_q)))
        super(RequirementValueFunction, self).__init__(t_q, t_req, 0, t_a, t_v)

    @property
    def gamma(self):
        """
        Get the gamma parameter.
        """
        return self._gamma


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    tv = RequirementValueFunction()
    tv.compile()
    q=np.linspace(0,2,100)
    plt.plot(q, tv(q, 1.))
    plt.xlabel('q')
    plt.ylabel('V')
    plt.legend()
    plt.show()

