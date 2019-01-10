"""
A class representing a simple 1D function.
Useful for compositions of functions.

"""


__all__ = ['Function']


import theano
from collections import Iterable


class Function(object):

    """
    A class representing a simple 1D function.
    Useful for compositions of functions.

    :param t_x:    List of theano variables. (immutable)
    :param t_f:    Theano variable that depends on t_x. (immutable)
    :param name:   A name for the function.
    """

    def __init__(self, t_x, t_f, name='Function'):
        if not isinstance(t_x, list):
            t_x = [t_x,]
        self._t_x = t_x
        self._t_f = t_f
        self._f_comp = None
        self._name = name

    @property
    def t_x(self):
        """
        Get t_x.
        """
        return self._t_x

    @property
    def t_f(self):
        """
        Get t_f.
        """
        return self._t_f

    @property
    def name(self):
        """
        Get the name of the function.
        """
        return self._name

    def compile(self):
        """
        Compiles the function.
        """
        if self._f_comp is None:
            self._f_comp = theano.function(self.t_x, self.t_f)

    def __call__(self, *x):
        """
        Evaluate the function at x.
        """
        if self._f_comp is None:
            raise RuntimeError('Fuction not compiled.')
        return self._f_comp(*x)

    def compose(self, g, t_x_elm=None):
        """
        Compose self with g.

        :param t_x_elm: The part of self.t_x that you want replaced with g.t_f.
        :param g:     Is a ``Function''
        :returns:       A ``Function'' representing f(g(x)).
        """
        if t_x_elm is None:
            t_x_elm = self.t_x[0]
        t_fg = theano.clone(self.t_f, replace={t_x_elm: g.t_f})
        return Function(g.t_x, t_fg)

    def grad(self, t_x_part):
        """
        Take the derivative of the function with respect to part of t_x.

        :param t_x_part:    Part of t_x (i.e., one of the items in the list).
        """
        t_f_grad = theano.grad(self.t_f, t_x_part)
        return Function(self.t_x, t_f_grad)

    def __str__(self):
        """
        Return a string representation of the object.
        """
        return self.__class__.__name__


if __name__ == '__main__':
    from theano import tensor as T
    import numpy as np
    t_x = T.dscalar('x')
    t_g = t_x ** 2
    g = Function(t_x, t_g)
    print 'g to str:', str(g)
    # This will fail because the function is not compiled
    try:
        g(2.)
    except Exception as e:
        print 'ERROR:', e
    g.compile()
    # This will work because the function is compiled
    print 'g(2.00) = %1.2f' % g(2.)
    # Now let's test the composition with another function
    t_x1 = T.dscalar('x1')
    t_f = t_x1 + 3.0
    f = Function(t_x1, t_f)
    f.compile()
    print 'f(4.00) = %1.2f' % f(4.)
    # Compose f with g
    fg = f.compose(g)
    fg.compile()
    print 'f(g(2.00)) = %1.2f' % fg(2.)

    # Let's test now the case of a function with two inputs (like the quality)
    t_e = T.dscalar('x')
    t_xi = T.dscalar('xi')
    t_q = 0.5 * t_e + 0.1 * t_xi
    q = Function([t_e, t_xi], t_q)
    q.compile()
    xi = np.random.randn()
    print 'q(0.4, %1.2f) = %1.2f' % (xi, q(0.5, xi))
    
    # Let's test the gradient of the function
    dqde = q.grad(t_xi)
    dqde.compile()
    print 'dq/de(0.4, %1.2f) = %1.2f' % (xi, dqde(0.5, xi))
