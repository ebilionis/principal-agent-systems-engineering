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
    agent_type11 = AgentType(LinearQualityFunction(1.6, 0.1),
                            QuadraticCostFunction(0.1),
                            ExponentialUtilityFunction(-2.0))

    agent_type12 = AgentType(LinearQualityFunction(1.6, 0.1),
                            QuadraticCostFunction(0.2),
                            ExponentialUtilityFunction(-2.0))
    agents = Agent([agent_type11, agent_type12])
    t = RequirementPlusIncentiveTransferFunction(gamma=30.)
    p = PrincipalProblem(ExponentialUtilityFunction(),
                        RequirementValueFunction(1, gamma=10.),
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
        type_1 = a[:4]
        type_2 = a[4:]

        g[0] += [g[0][0] - p._irc[0][0].evaluate(type_2)['exp_u_pi_e_star']]
        g[0] += [g[0][1] - p._irc[0][1].evaluate(type_1)['exp_u_pi_e_star']]

        return np.hstack([[f], g[0]])

    #@pm.deterministic
    #def results(a=a):
    #    return p.evaluate(a)
    
    @pm.stochastic(dtype=float, observed=True)
    def loglike(value=1.0, fg=fg, gamma=gamma):
        f = fg[0]
        g = fg[1]
        return gamma * f + \
                np.sum(np.log(1.0 / (1.0 + np.exp(- gamma * g))))
    return locals()


if __name__ == '__main__':
    model = make_model()
    mcmc = pm.MCMC(model)
    mcmc.use_step_method(ps.RandomWalk, model['a'], proposal_sd=0.01)
    smc = SteadyPaceSMC(mcmc, num_particles=100, num_mcmc=100, verbose=4,
                 gamma_is_an_exponent=True,
                 ess_reduction=0.9, adapt_proposal_step=True,
                 mpi=mpi)
    smc.initialize(0.1)
    for gamma in np.linspace(0.1, 10, 10)[1:]:
        smc.move_to(gamma)
        pa = smc.get_particle_approximation().gather()
        if mpi.COMM_WORLD.Get_rank() == 0:
            idx = np.argmax(pa.fg[:, 0])
            print('max f = ', pa.fg[idx, 0], 'g = ', pa.fg[idx, 1:])
            g = pa.fg[idx, 1:]
            print('\tType 1\tType 2')
            g00 = g[0]
            g01 = -(g[2] - g00)
            g11 = g[1]
            g10 = -(g[3] - g11)
            print('Type 1\t%1.3f\t%1.3f' % (g00, g01))
            print('Type 2\t%1.3f\t%1.3f' % (g10, g11))
            print('> ', pa.a[idx, :])

    if mpi.COMM_WORLD.Get_rank() == 0:
        idx = np.argmax(pa.fg[:, 0])
        print()
        print('*'*30,' final results ','*'*30)
        print()
        #print('final results: ', pa.results[idx])
        print('parameters: ', pa.a[idx, :])
        print()
        print('max f = ', pa.fg[idx, 0], 'g = ', pa.fg[idx, 1:])
        print('> ', pa.a[idx, :])
        print()
        print('*'*80)

# final results:  {'exp_u_pi_agents': [[0.12127722014621606, 0.03182667340882685]], 'exp_u_pi_0': array(0.91526288), 'e_stars': [1.0, 0.7664545374354419], 'd_exp_u_pi_0_da': array([-0.50000002, -0.49434743,  0.05082953, -0.13379131, -0.5       ,
#        -0.52885262,  0.02209308, -0.35815279])}
# parameters:  [0.01039644 0.06288117 1.33182003 0.08571926 0.00903602 0.03766459
#  0.93188156 0.08954284]

# max f =  0.9152628762549916 g =  [ 0.12127722  0.03182667 -0.02263952  0.00802762]
# >  [0.01039644 0.06288117 1.33182003 0.08571926 0.00903602 0.03766459
#  0.93188156 0.08954284]




