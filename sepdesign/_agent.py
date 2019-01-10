"""
A class representing an agent that could be of several types.

"""


__all__ = ['Agent']


from _types import AgentType
import numpy as np


class Agent(object):

    """
    A class representing an agent that could be of several types.

    :param agent_types:         A list of agent type (immutable).
    :param type_probabilities:  A probability for each type. Positive numbers
                                that sum to one. If ``None`` then a uniform
                                probability distribution is assumed (immutable).
    """

    def __init__(self, agent_types, type_probabilities=None):
        if isinstance(agent_types, AgentType):
            agent_types = [agent_types]
        assert isinstance(agent_types, list)
        for at in agent_types:
            assert isinstance(at, AgentType)
        self._agent_types = agent_types
        if type_probabilities is None:
            type_probabilities = np.ones(self.num_types)
        assert np.all(type_probabilities >= 0.)
        type_probabilities /= np.sum(type_probabilities)
        self._type_probabilities = type_probabilities

    @property
    def num_types(self):
        return len(self.agent_types)

    @property
    def agent_types(self):
        """
        Get the agent types.
        """
        return self._agent_types

    @property
    def type_probabilities(self):
        """
        Get the probability of each type.
        """
        return self._type_probabilities

    def __repr__(self):
        """
        Return a string representation of the object.
        """
        return 'Agent(agent_types=' + str(self.agent_types) +\
                ', type_probabilities=' + str(self.type_probabilities) + ')'


if __name__ == '__main__':
    from _quality_functions import *
    from _cost_functions import *
    from _utility_functions import *

    # Create an agent of a specific type
    agent_type = AgentType(LinearQualityFunction(1.5, 0.2),
                           QuadraticCostFunction(0.1),
                           ExponentialUtilityFunction(2.0))
    agent = Agent(agent_type)
    print str(agent)

    # Let's create more types
    agent_type2 = AgentType(LinearQualityFunction(2.5, 0.1),
                            QuadraticCostFunction(0.3),
                            ExponentialUtilityFunction(1.5))
    agent2 = Agent([agent_type, agent_type2])
    print str(agent2)
