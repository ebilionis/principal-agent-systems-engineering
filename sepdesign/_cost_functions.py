"""
Several cost functions that can be used by agents.

"""


__all__ = ['CostFunction', 'LinearCostFunction']


import theano
import theano.tensor as T
from _function import Function


class CostFunction(Function):
    """
    An abstract cost function class.
    
    This is a function of effort only t_e.
    Do not initialize this class.
    You should only initialize its descendants.
    """

    @property
    def t_e(self):
        """
        Get the symbolic variable for the effort.
        """
        return self.t_x[0]

    @property
    def t_c(self):
        """
        Get the symbolic variable for the cost.
        """
        return self.t_f


class LinearCostFunction(CostFunction):
    """
    A linear cost function.
    
    :param coef:    The value of the linear coefficient (cannot be changed).
    :param t_e:     The effort symbolic variable. Created from scratch if not
                    specified.
    """

    def __init__(self, coef, t_e=None):
        if t_e is None:
            t_e = T.dscalar('e')
        self._coef = coef
        t_c = coef * t_e
        super(LinearCostFunction, self).__init__(t_e, t_c)

    @property
    def coef(self):
        """
        Get the coefficient of the linear function.
        """
        return self._coef


if __name__ == '__main__':
    c = LinearCostFunction(0.5)
    c.compile()
    print 'c(0.5) = %1.2f' % c(0.5)