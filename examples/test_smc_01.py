"""
Test the optimization using SMC.

"""


from sepdesign import *
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import pysmc as ps
import mpi4py.MPI as mpi


class SteadyPaceSMC(ps.SMC):

    def _find_next_gamma(self, gamma):
        self._loglike = self._get_loglike(self.gamma, gamma)
        return gamma


def make_model():
    agent_type11 = AgentType(LinearQualityFunction(1.5, 0.2),
                            QuadraticCostFunction(0.1),
                            ExponentialUtilityFunction())
    agents = Agent([agent_type11])
    t = RequirementPlusIncentiveTransferFunction(gamma=30.)
    p = PrincipalProblem(ExponentialUtilityFunction(),
                        RequirementValueFunction(1, gamma=50.),
                        agents, t)
    p.compile()
    gamma = 1.0
    kappa = 10.0
    
    a = pm.Uniform('a', 0.0, 1.0, size=(p.num_param,))

    @pm.deterministic
    def fg(a=a):
        res = p.evaluate(a)
        # The thing that you want to maximize
        f = res['exp_u_pi_0']
        # The constraints that must be positive
        g = res['exp_u_pi_agents']
        return np.hstack([[f], g[0]])
    
    @pm.stochastic(dtype=float, observed=True)
    def loglike(value=1.0, fg=fg, gamma=gamma):
        f = fg[0]
        g = fg[1]
        return gamma * f + \
                np.sum(np.log(1.0 / (1.0 + np.exp(-gamma * g))))
    return locals()


if __name__ == '__main__':
    model = make_model()
    mcmc = pm.MCMC(model)
    mcmc.use_step_method(ps.RandomWalk, model['a'])
    smc = SteadyPaceSMC(mcmc, num_particles=4000, num_mcmc=1, verbose=4,
                 gamma_is_an_exponent=True,
                 ess_reduction=0.6, adapt_proposal_step=True,
                 mpi=mpi)
    smc.initialize(0.01)
    for gamma in np.linspace(0.0, 10.0, 100)[1:]:
        smc.move_to(gamma)
        pa = smc.get_particle_approximation().gather()
        if mpi.COMM_WORLD.Get_rank() == 0:
            idx = np.argmax(pa.fg[:, 0])
            print('max f = ', pa.fg[idx, 0], 'g = ', pa.fg[idx, 1:])
            print('> ', pa.a[idx, :])
