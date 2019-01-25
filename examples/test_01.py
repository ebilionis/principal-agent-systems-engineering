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
import pickle
from pyDOE import lhs
import numpy as np
from mpi4py import MPI


num_restarts = 1000
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

def split(container, count):
    """
    split the jobs for parallel run
    """
    return [container[_i::count] for _i in range(count)]

if rank == 0:
    agent_type11 = AgentType(LinearQualityFunction(1.2, 0.1),
                            QuadraticCostFunction(0.1),
                            ExponentialUtilityFunction())
    
    agents = Agent([agent_type11])
    
    t = RequirementPlusIncentiveTransferFunction(gamma=40.)

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

# with three runs and gamma = 20.0
#[{'x': array([0.00883673, 0.07939584, 1.35337989, 0.21942791]), 'obj': 0.8911277301139346}]
#[{'x': array([0.00789353, 0.07994556, 1.35859862, 0.22912218]), 'obj': 0.8920417224775055}]
#[{'x': array([0.00703105, 0.08080326, 1.36224673, 0.24160272]), 'obj': 0.8917897112483046}]

#with gamma = 40.0
#[{'x': array([6.20071441e-06, 2.45366832e-02, 1.02885840e+00, 1.60172500e-01]), 'obj': 0.8999919364536335}]





