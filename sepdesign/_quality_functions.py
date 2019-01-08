"""
Definitions of quality functions.

"""


__all__ = ['QualityFunction', 'LinearQualityFunction']


import theano
import theano.tensor as T
from _function import Function


class QualityFunction(Function):
    """
    An abstract class for quality function.

    Do not initialize it.
    """

    def __init__(self, t_e, t_xi, t_q):
        super(QualityFunction, self).__init__([t_e, t_xi], t_q)


    @property
    def t_e(self):
        """
        Get the symbolic variable for effort.
        """
        return self.t_x[0]

    @property
    def t_xi(self):
        """
        Get the symbolic variable for the state of nature.
        """
        return self.t_x[1]

    @property
    def t_q(self):
        """
        Get the symbolic variable corresponding to the quality function.
        """
        return self.t_f


class LinearQualityFunction(QualityFunction):
    """
    A linear quality function.

    :param e_coef:  The effort coefficient (immutable)
    :param xi_coef: The coefficient of the linear paramter (immutable).
    :param t_e:     The effort symbolic variable (initialized if None).
    :param t_xi:    The state of nature symbolic variable (initialized if None).
    """

    def __init__(self, e_coef, xi_coef, t_e=None, t_xi=None):
        if t_e is None:
            t_e = T.dscalar('e')
        if t_xi is None:
            t_xi = T.dscalar('xi')
        t_q = e_coef * t_e + xi_coef * t_xi
        super(LinearQualityFunction, self).__init__(t_e, t_xi, t_q)


if __name__ == '__main__':
    import numpy as np
    q = LinearQualityFunction(0.2, 0.1)
    q.compile()
    e = 0.5
    xi = np.random.randn()
    print 'q(%1.2f, %1.2f) = %1.2f' % (e, xi, q(e, xi))
