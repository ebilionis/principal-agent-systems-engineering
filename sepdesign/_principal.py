"""
This class includes the principal optimization problem.
"""
__all__ = ["PrincipalProblem"]

import theano
import theano.tensor as T
from _types import AgentType
from _transfer_functions import TransferFunction

class PrincipalProblem(object):
    """
    The PrincipalClass
    """
    def __init__(self, num_agents, agent_types, type_probs, transfer_func):
        assert len(agent_types) == num_agents
        assert len(type_probs)  == num_agents
        for n in range(num_agents):
            assert len(agent_types[n]) == len(type_probs[n])
            for a in agent_types[n]:
                assert isinstance(a, AgentType)
        assert isinstance(transfer_func, TransferFunction)
        self._agent_types   = agent_types
        self._type_probs    = type_probs
        self._transfer_func = transfer_func
        self.res            = [{} for x in range(num_agents)]

    def evaluate(self, parameters):
        """
        Evaluate the individual rationality constraints at specific
        transfer function parameters.

        :param a:   The parameters of the transfer function.
        """
        for n in range(num_agents):
            for t in range(len(self._agent_types[n])):
                res[t] = self._agent_types[n][t].evaluate(parameters)
                
                """
                .....
                .....
                .....
                I will continue from here
                """







if __name__ == "__main__":
    
