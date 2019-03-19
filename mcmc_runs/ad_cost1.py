"""
Test the optimization using SMC.

"""


from sepdesign import *
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import pysmc as ps
import pickle
import mpi4py.MPI as mpi


class SteadyPaceSMC(ps.SMC):

    pass
    #def _find_next_gamma(self, gamma):
    #    self._loglike = self._get_loglike(self.gamma, gamma)
    #    return gamma


def make_model():
    agent_type11 = AgentType(LinearQualityFunction(2.5, 0.4),
                            QuadraticCostFunction(0.1),
                            ExponentialUtilityFunction(-2.0))

    agent_type12 = AgentType(LinearQualityFunction(2.5, 0.4),
                            QuadraticCostFunction(0.4),
                            ExponentialUtilityFunction(-2.0))
    agent1 = Agent([agent_type11])
    agent2 = Agent([agent_type12])
    t = RequirementTransferFunction(gamma=50.)
    p1 = PrincipalProblem(ExponentialUtilityFunction(),
                        RequirementValueFunction(1, gamma=100.),
                        agent1, t)
    p2 = PrincipalProblem(ExponentialUtilityFunction(),
                        RequirementValueFunction(1, gamma=100.),
                        agent2, t)
    p1.compile()
    p2.compile()

    gamma = 1.0
    kappa = 10.0
    a = pm.Uniform('a', 0.0, 1.7, size=(3,))

    @pm.deterministic
    def fg(a=a):
        res1 = p1.evaluate(a)
        res2 = p2.evaluate(a)
        # The thing that you want to maximize
        f = 0.5*res1['exp_u_pi_0'] + 0.5*res2['exp_u_pi_0']
        # The constraints that must be positive
        g    =  res1['exp_u_pi_agents']
        g[0] += res2['exp_u_pi_agents'][0]

        return np.hstack([[f], g[0]])

    #@pm.deterministic
    #def results(a=a):
    #    return p.evaluate(a)
    
    @pm.stochastic(dtype=float, observed=True)
    def loglike(value=1.0, fg=fg, gamma=gamma):
        f = fg[0]
        g = fg[1:]
        # return gamma*f + gamma*20.0*(min(0., g[0]) + min(0., g[1]))
        return gamma * f + \
                np.sum(np.log(1.0 / (1.0 + np.exp(-gamma * g))))
    return locals()


if __name__ == '__main__':
    model = make_model()
    mcmc = pm.MCMC(model)
    mcmc.use_step_method(ps.RandomWalk, model['a'], proposal_sd=0.001)
    smc = SteadyPaceSMC(mcmc, num_particles=100, num_mcmc=2, verbose=4,
                 gamma_is_an_exponent=True,
                 ess_reduction=0.9, adapt_proposal_step=True,
                 mpi=mpi)
    smc.initialize(0.1)
    results = []
    for gamma in np.linspace(0., 40, 100)[1:]:
        smc.move_to(gamma)
        pa = smc.get_particle_approximation().gather()
        if mpi.COMM_WORLD.Get_rank() == 0:
            # idx = np.argmax(pa.fg[:, 0])
            idx = np.argmax(pa.weights)
            print('max f = ', pa.fg[idx, 0], 'g = ', pa.fg[idx, 1:])
            # g = pa.fg[idx, 1:]
            # print('\tType 1\tType 2')
            # g00 = g[0]
            # g01 = -(g[2] - g00)
            # g11 = g[1]
            # g10 = -(g[3] - g11)
            # print('Type 1\t%1.3f\t%1.3f' % (g00, g01))
            # print('Type 2\t%1.3f\t%1.3f' % (g10, g11))
            print('> ', pa.a[idx, :])
            results += [[pa.fg[idx, 0], pa.fg[idx, 1:], pa.a[idx, :]]]
    if mpi.COMM_WORLD.Get_rank() == 0:
        temp = [results[i][0]for i in range(len(results))]
        idx = np.argmax(temp)
        print(results[idx])

# max f =  0.6956473585700186 g =  [0.37573252 0.03126238]
# >  [0.00286624 0.30889572 1.08027044]