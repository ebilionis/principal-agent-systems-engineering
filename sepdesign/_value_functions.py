"""
Class of value functions
"""

__all__ = ['ValueFunction', 'RequirementValueFunction', 
           'RequirementPlusValueFunction']


import theano
import theano.tensor as T
import numpy as np
from sepdesign._function import Function


class ValueFunction(Function):
    """
    This is an abstract class for value functions.

    :param t_q: A list of ``T.dvector'''s.
    """
    def __init__(self, t_q, t_v):
        super(ValueFunction, self).__init__(t_q, t_v)


class RequirementValueFunction(ValueFunction):
    """
    This is a 0 or 1 value function.

    :param num_subsystems:  The number of subsystems that must meet the
                            requirements.
    """
    def __init__(self, num_subsystems, gamma=50.0):
        self._gamma = gamma
        self._num_subsystems = num_subsystems
        t_qs = [T.dvector('q%d' % i) for i in range(num_subsystems)]
        t_v = T.prod([1.0 / (1.0 + T.exp(gamma * (.5 - t_q))) 
                      for t_q in t_qs], axis=0)
        super(RequirementValueFunction, self).__init__(t_qs, t_v)

    @property
    def num_subsystems(self):
        """
        Get the subsystems.
        """
        return self._num_subsystems

    @property
    def gamma(self):
        """
        Get the gamma parameter.
        """
        return self._gamma


class RequirementPlusValueFunction(ValueFunction):
    """
    This is the value function that is growing by quality for more than 1.
    :param num_subsystems:  The number of subsystems that must meet the
                            requirements.
    :param alphas: The grow factor of the value function for each subsytem
    """   
    def __init__(self, num_subsystems, alphas, gamma=50.0):
        self._gamma = gamma
        self._num_subsystems = num_subsystems
        self._alphas = alphas
        t_qs = [T.dvector('q%d' % i) for i in range(num_subsystems)]
        t_v = T.prod([1.0 / (1.0 + T.exp(gamma * (1.0 - t_q))) * (ai * T.tanh(t_q - 1.0) + 1.0)
                      for t_q, ai in zip(t_qs, self._alphas)], axis=0)
        super(RequirementPlusValueFunction, self).__init__(t_qs, t_v)

    @property
    def num_subsystems(self):
        """
        Get the subsystems.
        """
        return self._num_subsystems

    @property
    def gamma(self):
        """
        Get the gamma parameter.
        """
        return self._gamma
    @property
    def alphas(self):
        """
        Get the alpha parameters.
        """
        return self._alphas



if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('white')
    sns.set_context('paper')

    # A value function for the case N = 1
    v1 = RequirementPlusValueFunction(1,[0.2])
    v1.compile()
    qs = np.linspace(0, 2, 100)
    v1s = np.array([v1([q]) for q in qs]) # the [q] is required because it
                                          # it works with vectors. The only
                                          # I want the vectors is for the
                                          # expectation over xi to be easy.
    fig, ax = plt.subplots()
    ax.plot(qs, v1s)
    ax.set_xlabel('q')
    ax.set_ylabel('V')

    # A value function for the case N = 2
    v2 = RequirementPlusValueFunction(2, np.array([0.2,0.3]))
    v2.compile()
    qs1 = np.linspace(0, 2, 100)
    qs2 = np.linspace(0, 2, 100)
    Q1, Q2 = np.meshgrid(qs1, qs2)
    V2s = np.array([[v2([Q1[i, j]], [Q2[i,j]])[0] for j in range(Q1.shape[1])]
                    for i in range(Q1.shape[0])])
    fig, ax = plt.subplots()
    c = ax.contourf(Q1, Q2, V2s)
    ax.set_xlabel('$q_1$')
    ax.set_ylabel('$q_2$')
    plt.colorbar(c)
    

