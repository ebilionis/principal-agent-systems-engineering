"""
Related to the types of agents.

"""


__all__ = ['AgentType']


import theano
import theano.tensor as T
import numpy as np
from sepdesign._function import Function
from sepdesign._quality_functions import QualityFunction
from sepdesign._cost_functions import CostFunction
from sepdesign._utility_functions import UtilityFunction
from sepdesign._transfer_functions import TransferFunction


class AgentType(object):
    """
    A class representing the type of an agent.
    
    :param q:       The quality function (immutable).
    :param c:       The cost function (immutable).
    :param u:       The utility function (immutable).
    :param name:    Added a name for the agent type.
    """

    def __init__(self, q, c, u, name='AgentType'):
        assert isinstance(q, QualityFunction)
        assert isinstance(c, CostFunction)
        assert isinstance(u, UtilityFunction)
        self._q = q
        self._c = c
        self._u = u
        self._name = name

    @property
    def q(self):
        """
        Get the quality function.
        """
        return self._q

    @property
    def c(self):
        """
        Get the cost function.
        """
        return self._c

    @property
    def u(self):
        """
        Get the utility function.
        """
        return self._u

    @property
    def name(self):
        """
        Get the name of this type.
        """
        return self._name

    def get_pi(self, t):
        """
        Get the payoff function of an agent.
        We will just return a Function of the right form.

        :param t:   A transfer function.
        """
        assert isinstance(t, TransferFunction)
        # Extract symbolic variables from quality
        t_e = self.q.t_e
        t_xi = self.q.t_xi
        t_q = self.q.t_q
        # Make sure t_q and t_c use the same t_e:
        t_c = theano.clone(self.c.t_c, replace={self.c.t_e: t_e})
        # Compose t_t with t_q
        t_t_comp_q = theano.clone(t.t_t, replace={t.t_q: t_q})
        # The parameters of the transfer function
        t_a = t.t_a
        # The symbolic representation of the payoff
        t_pi = t_t_comp_q - t_c
        return Function([t_e, t_xi, t_a], t_pi)

    def get_util(self, t):
        """
        Get the utility function of an agent.
        We will just return a Function of the right form.

        :param t:   A transfer function.
        """
        return self.u.compose(self.get_pi(t))
        t_e = self.q.t_e
        t_xi = self.q.t_xi
        t_a = t.t_a
        pi = self.get_pi(t)
        t_u = theano.clone(self.u.t_util, replace={self.u.t_pi: pi.t_f})
        return Function([t_e, t_xi, t_a], t_pi)

    def _get_expectation(self, fun, degree=10000):
        """
        Get the expectation of the function ``fun``.

        :param fun: The function fun is an instance of a ``Function''.
                    The assumption is that fun.t_x == [t_e, t_xi, t_a] and that
                    we wish to integrate over t_xi which is a N(0,1) r.v.
                    We integrate using a numerical quadrature rule.
        :param degree:  Integrates exactly polynomials of 2 * degree - 1.
        """
        assert isinstance(fun, Function)
        # Get quadrature rule
        Z, v = np.polynomial.hermite.hermgauss(degree)
        Xi = Z * np.sqrt(2.0)
        w = v / np.sqrt(np.pi)
        # Symbolic quadrature weights
        t_e = fun.t_x[0]
        t_xi = fun.t_x[1]
        t_a = fun.t_x[2]
        t_fun = theano.clone(fun.t_f, replace={t_xi: Xi})
        t_exp_fun = T.dot(w, t_fun)
        return Function([t_e, t_a], t_exp_fun)

    def get_exp_pi(self, t, degree=100):
        """
        Get the expectation of the payoff.
        """
        return self._get_expectation(self.get_pi(t),degree=degree)

    def get_exp_util(self, t, degree=100):
        """
        Get the expected utility of the payoff.
        """
        return self._get_expectation(self.get_util(t), degree=degree)

    def __repr__(self):
        """
        Return a string representation of this object.
        """
        return 'AgentType(%s, %s, %s)' % (str(self.q), str(self.c), str(self.u))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('white')
    sns.set_context('paper')
    import numpy as np
    from ._quality_functions import *
    from ._cost_functions import *
    from ._utility_functions import *
    from ._transfer_functions import *

    # Create an agent of a specific type
    agent_type = AgentType(LinearQualityFunction(1.5, 0.2),
                           QuadraticCostFunction(0.1),
                           ExponentialUtilityFunction(2.0))
    print('AgentType to str:', str(agent_type))

    # Create a transfer function
    t = RequirementPlusIncentiveTransferFunction()

    # Get the payoff of the agent as a Function
    pi = agent_type.get_pi(t)
    # Get the expectation of the payoff
    exp_pi = agent_type.get_exp_pi(t) # Should be the same as pi because of the
                                      # linear quality function with zero mean
                                      # Gaussian noise.
    # Get the utility function
    u = agent_type.get_util(t)
    exp_u = agent_type.get_exp_util(t)
    # Compile all functions
    pi.compile()
    exp_pi.compile()
    u.compile()
    exp_u.compile()
    # Let's plot the payoff
    # Set the parameters of the transfer function
    a = [0.05, 0.3, 1., 0.0]
    # Set the random state of nature to a value
    xi = [0.0]
    es = np.linspace(0, 1, 100)
    pis = np.array([pi(e, xi, a) for e in es])
    exp_pis = np.array([exp_pi(e, a) for e in es])
    fig, ax = plt.subplots()
    ax.plot(es, pis)
    ax.plot(es, exp_pis, '--')
    ax.set_xlabel('$e_i$')
    ax.set_ylabel(r'$\Pi_i$')
    
    # Plot a few samples of the utility (for random xi)
    # I also compute the sampling average to compare to the expectation
    fig1, ax1 = plt.subplots()
    s_exp_us = np.zeros(es.shape)
    num_samples = 10 # Increase this number of test accuracy of quadrature rule
                     # *** BILIONIS TESTED THIS *** 
    for i in range(num_samples):
        xi = np.random.randn(1)
        us = np.array([u(e, xi, a) for e in es])
        s_exp_us += us[:, 0]
        if i <= 5:
            ax1.plot(es, us, lw=1, color=sns.color_palette()[1])
    s_exp_us /= (1.0 * num_samples)
    ax1.plot(es, s_exp_us, '--', lw=2, color=sns.color_palette()[0]) 
    exp_us = np.array([exp_u(e, a) for e in es])
    ax1.plot(es, exp_us, ':', lw=2, color=sns.color_palette()[2])
    ax1.set_xlabel('$e_i$')
    ax1.set_ylabel(r'$U_i(\Pi_i)$')
    plt.show()
