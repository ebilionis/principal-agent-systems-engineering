"""
This class includes the principal optimization problem.
"""
__all__ = ['PrincipalProblem']


import theano
import theano.tensor as T
from _types import AgentType
from _agent import Agent
from _value_functions import ValueFunction
from _transfer_functions import TransferFunction
from _individual_rationality import IndividualRationality


class PrincipalProblem(object):
    """
    A class representing the problem faced by the principal.

    :param agents:  A list of agents. Each element of the list is either.
    """

    def __init__(self, v, agents, t):
        assert isinstance(v, ValueFunction)
        self._v = v
        if isinstance(agents, Agent):
            agents = [agents]
        assert isinstance(agents, list)
        for a in agents:
            assert isinstance(a, Agent)
        self._agents   = agents
        assert isinstance(t, TransferFunction)
        self._t = t
        # Create individual rationality constraints - This is all we need
        irs = {}
        for a in self._agents:
            irs[a] = {}
            for at in a.agent_types:
                irs[a][at] = IndividualRationality(at, t)
        self._irs = irs

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

    # Create a value function
    v = RequirementValueFunction()

    # Create the principal's problem
    p = PrincipalProblem(v, agents, t)
    print str(p)
    print str(p._irs)

    # Test the evaluation of the system problem at a specific set of parameters
    a = [0.00, 0.3, 1., 0.1]
    
