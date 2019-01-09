"""
Everything related to the individual rationality constraints.

"""


__all__ = ['IndividualRationality']


import theano
import theano.tensor as T
import scipy.optimize as opt
from _types import AgentType
from _transfer_functions import TransferFunction


class IndividualRationality(object):
    """
    A class that facilitates the implementation of the individual rationality
    constraints.

    :param agent_type:          An instance of the class AgentType (immutable)
    :param transfer_function:   An instance of the class TransferFunction
                                (immutable)
    """

    def __init__(self, agent_type, transfer_func):
        assert isinstance(agent_type, AgentType)
        assert isinstance(transfer_func, TransferFunction)
        self._agent_type = agent_type
        self._transfer_func = transfer_func
        # Set up the optimization problem we need to solve
        # Get expected utility of payoff
        self.exp_u_pi = agent_type.get_exp_util(transfer_func)
        # Symbolic variable for effort
        t_e = self.exp_u_pi.t_x[0]
        # Symbolic variable for transfer function parameters
        t_a = self.exp_u_pi.t_x[1]
        # Get the gradient of the expected utility with respect to effort
        self.exp_u_pi_g_e = self.exp_u_pi.grad(t_e)
        # Get the second derivative of the expected utility wrt to effort
        self.exp_u_pi_g_e2 = self.exp_u_pi_g_e.grad(t_e)
        # We also need the mixed derivative of the exp. util. wrt to effort
        # and transfer parameters
        self.exp_u_pi_g_ea = self.exp_u_pi_g_e.grad(t_a)
        self._compiled = False

    def compile(self):
        """
        Compile everything that needs to be compiled.
        """
        if self._compiled:
            return
        for obj in [self.exp_u_pi, self.exp_u_pi_g_e, self.exp_u_pi_g_e2,
                    self.exp_u_pi_g_ea]:
            obj.compile()
        self._compiled = True
        # The objective function to be minimized
        self._obj_fun = lambda _e, _a: -self.exp_u_pi(_e[0], _a)
        self._obj_fun_jac = lambda _e, _a: -self.exp_u_pi_g_e(_e[0], _a)

    def evaluate(self, a, num_restarts=5):
        """
        Evaluate the individual rationality constraints at specific
        transfer function parameters.

        :param a:   The parameters of the transfer function.
        """
        if not self._compiled:
            raise RuntimeError('Compile before attempting to evaluate.')
        # Sanity check
        if not isinstance(a, np.ndarray):
            a = np.array(a)
        assert a.ndim == 1
        assert a.shape[0] == self.transfer_func.num_a
        # Restart points (excludes bounds)
        e0s = np.linspace(0, 1, num_restarts + 2)[1:-1]
        # Solve for each one of the restart points
        r_res = None
        r_min = 1e99
        all_opt_failed = True
        for e0 in e0s:
            res = opt.minimize(self._obj_fun, e0, args=a,
                               jac=self._obj_fun_jac, tol=1e-16,
                               method='SLSQP', bounds=((0.0, 1.0),))
            if res.success:
                all_opt_failed = False
                if r_min > res.fun:
                    r_res = res
        if all_opt_failed:
            raise RuntimeError('All the restarts failed.')
        e_star = res.x[0]
        exp_u_pi_e_star = -res.fun
        exp_u_pi_g_e_star = -res.jac
        exp_u_pi_g_e2_star = self.exp_u_pi_g_e2(e_star, a)
        exp_u_pi_g_ea_star = self.exp_u_pi_g_ea(e_star, a)
        if np.isclose(e_star, 0.0):
            # Left constraint is active
            # g(e) = -e <= 0 -> dg/de = -1
            print 'HERE'
            print 'e_star:', e_star
            #A = np.array([[exp_u_pi_g_e2_star - exp_u_pi_g_e_star, 1.0],
            #               1.0, 0.0])
            #b = np.array([[-exp_u_pi_g_ea_star[None, :] + exp_u_pi_g_e_star],
            #               np.zeros((1, a.shape[0]))])
            #print A
            #print b
            pass
        elif np.isclose(e_star, 1.0):
            # Right constraint is active
            # g(e) = e - 1 <= 0 -> dg/de = 1
            e_star_g_a = None
            pass
        else:
            # Optimum is internal point
            e_star_g_a = -exp_u_pi_g_ea_star / exp_u_pi_g_e2_star
        e_star_g_a = -exp_u_pi_g_ea_star / exp_u_pi_g_e2_star
        res['e_star'] = e_star
        res['e_star_g_a'] = e_star_g_a
        res['exp_u_pi_e_star'] = exp_u_pi_e_star
        return res

    @property
    def agent_type(self):
        """
        Get agent type.
        """
        return self._agent_type

    @property
    def transfer_func(self):
        """
        Get the transfer function.
        """
        return self._transfer_func


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('white')
    sns.set_context('paper')
    import numpy as np
    import numdifftools as nd
    from _quality_functions import *
    from _cost_functions import *
    from _utility_functions import *
    from _transfer_functions import *

    # Create an agent of a specific type
    agent_type = AgentType(LinearQualityFunction(1.5, 0.2),
                           QuadraticCostFunction(0.),
                           ExponentialUtilityFunction(2.0))

    # Create a transfer function
    t = RequirementPlusIncentiveTransferFunction()
    
    # Create the individual rationality constraint for this person
    ir = IndividualRationality(agent_type, t)
    
    # Compile everything we need to solve the individual rationality const.
    ir.compile()

    # Evaluate the individual rationality constraints at specific transfer
    # function parameters
    a = [0.05, 0.3, 1., 0.1]
    res = ir.evaluate(a)
    # The optimal effort is here
    e_star = res['e_star']
    print 'e_star = %1.2f' % e_star
    # The expected utility at the optimal effort
    exp_u_pi_e_star = res['exp_u_pi_e_star']
    print 'E_i[U_i(Pi_i(e_star; a))] = %1.2f' % exp_u_pi_e_star
    # Let's compare e_star_g_e to the numerical derivative
    e_star_g_a = res['e_star_g_a']
    func = lambda _a: ir.evaluate(_a)['e_star']
    func_g_a = nd.Gradient(func)
    n_e_star_g_a = func_g_a(a)
    print 'e_star_g_a = ', e_star_g_a
    print 'n_e_star_g_a =', n_e_star_g_a
    print 'Close?', np.allclose(e_star_g_a, n_e_star_g_a, atol=1e-3)
