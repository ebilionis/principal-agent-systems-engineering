"""
This class includes the principal optimization problem.
"""
__all__ = ['PrincipalProblem']


import theano
import theano.tensor as T
import itertools
import design
import numpy as np
from _types import AgentType
from _agent import Agent
from _utility_functions import UtilityFunction
from _value_functions import ValueFunction
from _transfer_functions import TransferFunction
from _individual_rationality import IndividualRationality


class PrincipalProblem(object):
    """
    A class representing the problem faced by the principal.

    :param u:           A utility function for the principal.
    :param v:           A value function for the system.
    :param agents:      A list of agents. Each element of the list is either.
    :param t:           The form of the transfer function.
    :param sg_level:    The sparse grid level for taking the expectation over
                        the xi's.
    """

    def __init__(self, u, v, agents, t, sg_level=5):
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
        # Individual rationality constraints (i-k)
        irs = [[] for _ in range(self.num_agents)]
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
                irs[i].append(
                        IndividualRationality(self.agents[i].agent_types[k], t))
                t_as[i].append(T.dvector('a{%d,%d}' % (i, k)))
                t_e_stars[i].append(T.scalar('e_stars{%d,%d}' % (i, k)))
                q_base = self.agents[i].agent_types[k].q
                t_qs[i].append(theano.clone(q_base.t_f,
                    replace={q_base.t_x[0]: t_e_stars[i][k],
                             q_base.t_x[1]: t_xis[i]}))
                t_ts[i].append(theano.clone(t.t_f,
            replace={t.t_x[0]: t_qs[i][k],
                     t.t_x[1]: t_as[i][k]}))
        self._irs = irs
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
            t_u = theano.clone(u.t_f, replace={u.t_x[0]: t_pi_comb})
            # Start summing up
            t_sum_u_over_comb += p_comb * t_u
        #theano.printing.pydotprint(t_sum_u_over_comb, outfile='tmp.png')
        # Take the expectation over the Xi's numerically
        Z, w_unorm = design.sparse_grid(self.num_agents, sg_level, 'GH')
        Xi = Z * np.sqrt(2.0)
        w = w_unorm / np.sqrt(np.pi ** self.num_agents)
        t_tmp = theano.clone(t_sum_u_over_comb,
                    replace=dict((t_xis[i], Xi[:, i])
                                 for i in range(self.num_agents)))
        #theano.printing.pydotprint(t_tmp, outfile='tmp.png')
        # THEANO OBJECT REPRESENTING THE EXPECTED UTILITY OF THE PRINCIPAL:
        t_exp_u = T.dot(w, t_tmp)

    def _agent_type_range(self):
        """
        Returns an iterator over all possible combinations of agent types.
        """
        return itertools.product(*(range(a.num_types) for a in self.agents))
    
    @property
    def num_agents(self):
        """
        Get the number of agents.
        """
        return len(self.agents)

    @property
    def compile(self):
        """
        Compile everything we need to evaluate the objective and constraints.
        """
        for a in self._agents:
            for at in a.agent_types:
                irs[a][at].compile()

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
    print str(p)
    print str(p._irs)

    # Test looping over agent type combinations
    for comb in p._agent_type_range():
        print comb

    # Test the evaluation of the system problem at a specific set of parameters
    a = [0.00, 0.3, 1., 0.1]
    
