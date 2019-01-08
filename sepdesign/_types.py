"""
Related to the types of agents.

"""


__all__ = ['AgentType']


import theano
import theano.tensor as T
import numpy as np
from _function import Function
from _quality_functions import QualityFunction
from _cost_functions import CostFunction
# from _utility_functions import UtilityFunction # TODO: Remove comment


class AgentType(object):
    """
    A class representing the type of an agent.
    
    :param q:  The quality function (immutable).
    :param c:  The cost function (immutable).
    :param u:  The utility function (immutable).
    """

    def __init__(self, q, c, u):
        assert isinstance(q, QualityFunction)
        assert isinstance(c, CostFunction)
        #assert isinstance(u, UtilityFunction) # TODO: Remove comment
        self.q = q
        self.c = c
        self.u = u

    def get_pi(self, t):
        """
        Get the payoff function of an agent.
        We will just return a Function of the right form.

        :param t:   A transfer function.
        """
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

    def _get_expectation(self, fun, degree=100):
        """
        Get the expectation of the function ``fun``.

        :param fun: The function fun is an instance of a ``Function''.
                    The assumption is that fun.t_x == [t_e, t_xi, t_a] and that
                    we wish to integrate over t_xi which is a N(0,1) r.v.
                    We integrate using a numerical quadrature rule.
        :param degree:  Integrates exactly polynomials of 2 * degree - 1.
        """
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('white')
    sns.set_context('paper')
    import numpy as np
    from _quality_functions import *
    from _cost_functions import *
    from _transfer_functions import *

    # Create an agent of a specific type
    agent_type = AgentType(LinearQualityFunction(1.5, 0.01),
                           LinearCostFunction(0.1),
                           None) # TODO: Fix when utility ready

    # Create a transfer function
    t = RequirementPlusIncentiveTransferFunction()

    # Get the payoff of the agent as a Function
    pi = agent_type.get_pi(t)
    exp_pi = agent_type.get_exp_pi(t) # Should be the same as pi because of the
                                      # linear quality function with zero mean
                                      # Gaussian noise.
    # Compile it if you want to evaluate it as a function
    pi.compile()
    exp_pi.compile()
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
    ax.set_ylabel(r'$\Pi_i(e_i) | \Xi_i=0$')
    plt.show()
