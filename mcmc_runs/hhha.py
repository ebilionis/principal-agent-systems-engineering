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

    def _find_next_gamma(self, gamma):
        self._loglike = self._get_loglike(self.gamma, gamma)
        return gamma


def make_model():
    agent_type11 = AgentType(LinearQualityFunction(1.6, 0.3),
                            QuadraticCostFunction(0.1),
                            ExponentialUtilityFunction(-2.0))

    agent_type12 = AgentType(LinearQualityFunction(2.0, 0.1),
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
    a = pm.Uniform('a', 0.0, 1.5, size=(p.num_param,))

    @pm.deterministic
    def fg(a=a):
        res = p.evaluate(a)
        # The thing that you want to maximize
        f = res['exp_u_pi_0']
        # The constraints that must be positive
        g = res['exp_u_pi_agents']

        # g[0] += [p._irc[0][0].evaluate(a[:4])['exp_u_pi_e_star']-\
        #       p._irc[0][0].evaluate(a[4:])['exp_u_pi_e_star']]

        # g[0] += [p._irc[0][1].evaluate(a[4:])['exp_u_pi_e_star']-\
        #       p._irc[0][1].evaluate(a[:4])['exp_u_pi_e_star']]


        return np.hstack([[f], g[0]])
    @pm.deterministic
    def results(a=a):
        return p.evaluate(a)
    
    @pm.stochastic(dtype=float, observed=True)
    def loglike(value=1.0, fg=fg, gamma=gamma):
        f = fg[0]
        g = fg[1]
        return gamma * f + \
                np.sum(np.log(1.0 / (1.0 + np.exp(-gamma*5. * g))))
    return locals()


if __name__ == '__main__':
    model = make_model()
    mcmc = pm.MCMC(model)
    mcmc.use_step_method(ps.RandomWalk, model['a'])
    smc = SteadyPaceSMC(mcmc, num_particles=400, num_mcmc=1, verbose=4,
                 gamma_is_an_exponent=True,
                 ess_reduction=0.75, adapt_proposal_step=True,
                 mpi=mpi)
    smc.initialize(0.01)
    for gamma in np.linspace(.0, 50.0, 500)[1:]:
        smc.move_to(gamma)
        pa = smc.get_particle_approximation().gather()
        if mpi.COMM_WORLD.Get_rank() == 0:
            idx = np.argmax(pa.fg[:, 0])
            print('max f = ', pa.fg[idx, 0], 'g = ', pa.fg[idx, 1:])
            print('> ', pa.a[idx, :])

    if mpi.COMM_WORLD.Get_rank() == 0:
        idx = np.argmax(pa.fg[:, 0])
        print()
        print('*'*30,' final results ','*'*30)
        print()
        print('final results: ', pa.results[idx])
        print('parameters: ', pa.a[idx, :])
        print()
        print('max f = ', pa.fg[idx, 0], 'g = ', pa.fg[idx, 1:])
        print('> ', pa.a[idx, :])
        print()
        print('*'*80)

# final results:  {'exp_u_pi_agents': [[0.0111323320837401]], 'exp_u_pi_0': array(0.86974387), 'e_stars': [1.0], 'd_exp_u_pi_0_da': array([-0.99999998, -0.83950792,  0.13410894, -0.32671708])}
# parameters:  [8.27429785e-04 1.01691921e-01 1.29624561e+00 6.36276844e-02]



