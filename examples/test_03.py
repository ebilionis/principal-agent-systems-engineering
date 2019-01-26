import sys
sys.path.append('../sepdesign')
from _quality_functions import *
from _cost_functions import *
from _utility_functions import *
from _transfer_functions import *
from _value_functions import *
from _types import *
from _agent import *
from _principal import *
from _tools import *
import pickle
from pyDOE import lhs
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI


num_restarts = 2000
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

if rank == 0:
    agent_type11 = AgentType(LinearQualityFunction(1.5, 0.05),
                            QuadraticCostFunction(0.1),
                            ExponentialUtilityFunction())
    
    agents = Agent([agent_type11])
    
    t = RequirementPlusIncentiveTransferFunction(gamma=30.)

    p = PrincipalProblem(ExponentialUtilityFunction(),
                        RequirementValueFunction(1, gamma=50.),
                        agents, t)
    samples = lhs(4, num_restarts, 'c')
    # samples   = np.random.rand(num_restarts, 4)
    jobs = list(range(num_restarts))
    jobs = split(jobs, size)
else:
    agent_type11 = None
    agents       = None
    t            = None
    p            = None
    jobs         = None
    samples      = None

agent_type11 = comm.bcast(agent_type11, root=0)
agents       = comm.bcast(agents,       root=0)
t            = comm.bcast(t,            root=0)
p            = comm.bcast(p,            root=0)
p.compile()
samples      = comm.bcast(samples,      root=0)
jobs         = comm.scatter(jobs,       root=0)

results_all  = []
results_all  = [p.optimize_contract(num_restarts, samples, jobs)]
results_all = comm.gather(results_all, root = 0)
if rank == 0:
    print results_all
    final_result = results_all[np.argmax([results_all[i][0]['obj'] for i in range(size)])]
    print final_result

#results
#[{'x': array([0.0186288 , 0.05726905, 0.97723873, 0.07630824]), 'obj': 0.9039183855949247}]
