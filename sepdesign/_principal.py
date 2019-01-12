"""
This class includes the principal optimization problem.
"""
__all__ = ['PrincipalProblem']


import theano
import theano.tensor as T
import itertools
import design
import numpy as np
import sys
from _types import AgentType
from _agent import Agent
from _utility_functions import UtilityFunction
from _value_functions import ValueFunction
from _transfer_functions import TransferFunction
from _individual_rationality import IndividualRationality


# Flattens a list of lists
flatten = lambda l: [item for sublist in l for item in sublist]


def run_and_print(func, verbosity, *args, **kwargs):
    """
    Run func(*args, **kwargs) and print something msg.
    """
    if verbosity >= 0:
        sys.stdout.write('Running ' + str(func) + '...')
    res = func(*args, **kwargs)
    if verbosity >= 0:
        sys.stdout.write(' Done!\n')
    return res


class PrincipalProblem(object):
    """
    A class representing the problem faced by the principal.

    :param u:           A utility function for the principal.
    :param v:           A value function for the system.
    :param agents:      A list of agents. Each element of the list is either.
    :param t:           The form of the transfer function.
    :param sg_level:    The sparse grid level for taking the expectation over
                        the xi's.
    :param verbosity:   The verbosity level of the class.
    """

    def __init__(self, u, v, agents, t, sg_level=5, verbosity=1):
        assert isinstance(u, UtilityFunction)
        self._u = u
        assert isinstance(v, ValueFunction)
        self._v = v
        if isinstance(agents, Agent):
            agents = [agents]
        assert isinstance(agents, list)
        for a in agents:
            assert isinstance(a, Agent)
        assert len(agents) == v.num_subsystems
        self._agents = agents
        assert isinstance(t, TransferFunction)
        self._t = t
        assert isinstance(sg_level, int)
        self._sg_level = sg_level
        assert isinstance(verbosity, int)
        self._verbosity = verbosity
        self._setup_exp_u()
        self._num_param = np.sum([a.num_types for a in self.agents]) * t.num_a
        self._compiled = False

    def _setup_exp_u(self):
        """
        Set up the following:
        + self.exp_u_raw:   The expected utility of the principal as a
                            Function of e_star and the transfer function
                            parameters a. This is a Function. 
        + self.exp_u:       The expected utility of the principal as a function
                            of the transfer function parameters a. This is a
                            common Python function. It also returns the
                            gradient of the exp_u with respect to a.
        """
        # Setup the individual rationality constraints
        self._setup_irc()
        # Symbolic parameters of transfer functions (i-k)
        t_as = [[] for _ in range(self.num_agents)]
        # Symbolic optimal efforts (i-k)
        t_e_stars = [[] for _ in range(self.num_agents)]
        # Symbolic xis (i)
        t_xis = []
        # Symbolic efforts (i-k)
        t_qs = [[] for _ in range(self.num_agents)]
        t_ts = [[] for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            t_xis.append(T.dvector('xi{%d}' % i))
            for k in range(self.agents[i].num_types):
                t_as[i].append(T.dvector('a{%d,%d}' % (i, k)))
                t_e_stars[i].append(T.scalar('e_stars{%d,%d}' % (i, k)))
                q_base = self.agents[i].agent_types[k].q
                t_qs[i].append(theano.clone(q_base.t_f,
                    replace={q_base.t_x[0]: t_e_stars[i][k],
                             q_base.t_x[1]: t_xis[i]}))
                t_ts[i].append(theano.clone(t.t_f,
            replace={t.t_x[0]: t_qs[i][k],
                     t.t_x[1]: t_as[i][k]}))
        # For all possible combinations of agent types
        # Expected utility functions
        t_sum_u_over_comb = T.zeros((1,))
        for at_comb in self._agent_type_range(): # Loop over agent type combs
            # Get the value function for this combination of types in theano:
            t_v_comb = theano.clone(self.v.t_f,
                            replace=dict((self.v.t_x[i], t_qs[i][at_comb[i]])
                                         for i in range(self.num_agents)))
            # The payoff to the principal for this combination of types
            t_pi_comb = t_v_comb - T.sum(
                    [t_ts[i][at_comb[i]] for i in range(self.num_agents)], 
                                         axis=0)
            # Multiply with the probability of type happening
            p_comb = np.prod([self.agents[i].type_probabilities[at_comb[i]] 
                              for i in range(self.num_agents)])
            # The utility of the principal for this combination of types
            t_u = theano.clone(self.u.t_f, replace={self.u.t_x[0]: t_pi_comb})
            # Start summing up
            t_sum_u_over_comb += p_comb * t_u
        #theano.printing.pydotprint(t_sum_u_over_comb, outfile='tmp.png')
        # Take the expectation over the Xi's numerically
        Z, w_unorm = design.sparse_grid(self.num_agents, self._sg_level, 'GH')
        Xi = Z * np.sqrt(2.0)
        w = w_unorm / np.sqrt(np.pi ** self.num_agents)
        t_tmp = theano.clone(t_sum_u_over_comb,
                    replace=dict((t_xis[i], Xi[:, i])
                                 for i in range(self.num_agents)))
        #theano.printing.pydotprint(t_tmp, outfile='tmp.png')
        # THEANO OBJECT REPRESENTING THE EXPECTED UTILITY OF THE PRINCIPAL:
        t_exp_u_raw = T.dot(w, t_tmp)
        t_e_stars_f = flatten(t_e_stars)
        t_as_f = flatten(t_as)
        self._exp_u_raw = Function(t_e_stars_f + t_as_f, t_exp_u_raw)
        # Take derivative with respect to e_stars
        self._exp_u_raw_g_e = self._exp_u_raw.grad(t_e_stars_f)
        # Take derivative with respect to the as
        self._exp_u_raw_g_a = self._exp_u_raw.grad(t_as_f)

    def compile(self):
        """
        Compile all Functions.
        """
        run_and_print(self.exp_u_raw.compile, self.verbosity) 
        run_and_print(self.exp_u_raw_g_e.compile, self.verbosity)
        run_and_print(self.exp_u_raw_g_a.compile, self.verbosity)
        for i in range(self.num_agents):
            for k in range(self.agents[i].num_types):
                run_and_print(self._irc[i][k].compile, self.verbosity)
        self._compiled = True

    def evaluate(self, a):
        """
        Evaluate the expected utility of the principal along its gradient
        wrt to a.
        """
        if not self._compiled:
            raise RuntimeError('You must compile first.')
        # We will return a dictionary with the results
        res = {}
        # aas[i][k] is the transfer parameters of agent i type k
        aas = [[] for i in range(self.num_agents)]
        # e_stars[i][k] is the optimal effort of agent i type k
        e_stars = [[] for i in range(self.num_agents)]
        # e_stars_g_a[i][k] is the gradient of the optimal effort of agent i
        # type k with respect to aas[i][k]
        e_stars_g_a = [[] for i in range(self.num_agents)]
        # exp_u_pi_e_stars[i][k] is the expected utility of agent i type k
        # at e_stars[i][k] using transfer parameters aas[i][k]
        exp_u_pi_e_stars = [[] for i in range(self.num_agents)]
        count_as = 0
        for i in range(self.num_agents):
            ag_i = self.agents[i]
            a_i = a[count_as:count_as + self.t.num_a * ag_i.num_types]
            count_as += ag_i.num_types
            for k in range(ag_i.num_types):
                a_ik = a_i[k * self.t.num_a:(k+1) * self.t.num_a]
                aas[i].append(a_ik)
                res_ik = self._irc[i][k].evaluate(a_ik)
                e_stars[i].append(res_ik['e_star'])
                e_stars_g_a[i].append(res_ik['e_star_g_a'])
                exp_u_pi_e_stars[i].append(res_ik['exp_u_pi_e_star'])
        # Flatten the list in order to pass them to the functions
        e_stars_f = flatten(e_stars)
        aas_f = flatten(aas)
        # Evaluate the expected utility of the principal
        exp_u_pi_0 = self._exp_u_raw(*(e_stars_f + aas_f))
        res['exp_u_pi_0'] = exp_u_pi_0
        # Evaluate derivative of exp_u_pi_0 with respect to e at e_stars and a
        exp_u_pi_0_raw_g_e = self._exp_u_raw_g_e(*(e_stars_f + aas_f))
        # Evaluate derivative of exp_u_pi_0 with respect to a at e_stars and a
        exp_u_pi_0_raw_g_a = self._exp_u_raw_g_a(*(e_stars_f + aas_f))
        # TODO: SALAR
        # Populate res['exp_u_pi_0_g_a'] using the chain rule.
        return res

    def _setup_irc(self):
        """
        Set up individual rationality constraints.
        """
        # Individual rationality constraints (i-k)
        irc = [[] for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            for k in range(self.agents[i].num_types):
                irc[i].append(
                        IndividualRationality(self.agents[i].agent_types[k], t))
        self._irc = irc

    def _agent_type_range(self):
        """
        Returns an iterator over all possible combinations of agent types.
        """
        return itertools.product(*(range(a.num_types) for a in self.agents))

    @property
    def verbosity(self):
        """
        Return the verbosity level of the class.
        """
        return self._verbosity

    @property
    def exp_u_raw(self):
        """
        Get the expected utility of the principal as a Function with inputs
        e_star and the transfer function parameters a.
        """
        return self._exp_u_raw

    @property
    def exp_u_raw_g_e(self):
        """
        Return the derivative of the expected utility of the principal with
        respect to all e_stars as a function of e_star and the transfer
        function parameters a.
        """
        return self._exp_u_raw_g_e

    @property
    def exp_u_raw_g_a(self):
        """
        Return the derivative of the expected utility of the principal with
        respect to all transfer function parameters a as a function of e_star
        and a.
        """
        return self._exp_u_raw_g_a

    def d_exp_u_da(self, *x):
        """
        This is to calculate the total derivative of the expected utility of
        the principal w.r.t. parameters a.
        """
        self.d_exp_u_da_list = []
        if self._compiled is False:
            self.compile()
        num_agent_types = np.sum([a.num_types for a in self._agents])
        for i in range(num_agent_types):
            self.d_exp_u_da_list += [self.exp_u_raw_g_e(*x)[i] * self.exp_u_raw_g_a(*x)[i]]
        return self.d_exp_u_da_list



    #     return self.exp_u_raw_g_a
    
    @property
    def num_agents(self):
        """
        Get the number of agents.
        """
        return len(self.agents)

    @property
    def agents(self):
        """
        Get the agents.
        """
        return self._agents

    @property
    def t(self):
        """
        Get the transfer function.
        """
        return self._t

    @property
    def v(self):
        """
        Get the value function.
        """
        return self._v

    @property
    def u(self):
        """
        Get the utility function of the principal.
        """
        return self._u

    @property
    def num_param(self):
        """
        Get the total number of transfer function parameters.
        """
        return self._num_param

    def __repr__(self):
        """
        Return a string representation of the class.
        """
        return 'PrincipalProblem(v=' + str(self.v) + \
                ', agents=' + str(self.agents) + \
                ', t=' + str(self.t)


if __name__ == '__main__':
    from _quality_functions import *
    from _cost_functions import *
    from _utility_functions import *
    from _transfer_functions import *
    from _value_functions import *

    # Create an agent of a specific type
    agent_type11 = AgentType(LinearQualityFunction(1.5, 0.2),
                           QuadraticCostFunction(0.1),
                           ExponentialUtilityFunction(2.0))
    agent_type12 = AgentType(LinearQualityFunction(2.5, 0.1),
                            QuadraticCostFunction(0.3),
                            ExponentialUtilityFunction(1.5))
    agent_type21 = AgentType(LinearQualityFunction(2.5, 0.1),
                            QuadraticCostFunction(0.3),
                            ExponentialUtilityFunction(1.5))
    agent_type22 = AgentType(LinearQualityFunction(1.5, 0.3),
                            QuadraticCostFunction(0.1),
                            ExponentialUtilityFunction(0.0))

    # Create the agents
    agent1 = Agent([agent_type11, agent_type12])
    agent2 = Agent([agent_type21, agent_type22])
    agents = [agent1, agent2]

    # Create a transfer function
    t = RequirementPlusIncentiveTransferFunction()

    # Create the principal's problem
    p = PrincipalProblem(ExponentialUtilityFunction(0.0),
                         RequirementValueFunction(2),
                         agents, t)

    # Compile everything
    p.compile()

    # p.exp_u_raw_g_e.compile()

    #print p.d_exp_u_da(0.5, 0.5, 0.5, 0.5,[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1])

    # Test
    a = np.random.rand(p.num_param)
    p.evaluate(a)
