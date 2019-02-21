import sys
sys.path.append('../')
from src import *
import pickle
from mpi4py	import MPI
import numpy as np


number_opt = 20	# Set numer of restarts
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
if rank == 0:

	# Set N, q_coeff, ucoeff, delta, cs
	N                     = 1  									 # Number of agents
	q_coeff 			  = [0., 1.2] # Coefficients to define the quality function
	ucoeff                = 0.0									 # Utility coefficient (RA, RN, RP)
	delta                 = np.array([0.1])						 # Uncertainty in the quality
	cs                    = np.array([0.4])						 # The coefficient of the cost



	M                     = 4
	ncoloc                = 1000
	mu                    = np.linspace(1.0, 2.1, M)
	qvals                 = np.array([1.0])
	quads, weights, w_acc = roots_hermitenorm(ncoloc, mu=True)
	quads_bcast           = np.array([quads]*1).T
	weights               = weights.reshape(1, -1)
	others                = [weights, quads_bcast, w_acc]

	jobs = list(range(number_opt))
	jobs = split(jobs, size)
	np.random.seed(rank**2)

else:
	N           = None
	q_coeff    	= None	 
	ucoeff      = None
	delta       = None
	cs          = None
	M           = None
	ncoloc      = None
	mu          = None
	qvals       = None
	quads       = None
	weights     = None
	w_acc       = None
	quads_bcast = None
	others      = None
	sys         = None
	jobs        = None
	np.random.seed(rank**2)

results_all = []
jobs 		= comm.scatter(jobs, root=0)

N 			= comm.bcast(N, root = 0)
q_coeff 	= comm.bcast(q_coeff, root=0)
ucoeff  	= comm.bcast(ucoeff, root = 0)
delta   	= comm.bcast(delta, root = 0)
cs 			= comm.bcast(cs, root = 0)
M 			= comm.bcast(M, root = 0)
ncoloc 		= comm.bcast(ncoloc, root = 0)
mu 			= comm.bcast(mu, root = 0)
qvals 		= comm.bcast(qvals, root = 0)
quads 		= comm.bcast(quads, root = 0)
weights 	= comm.bcast(weights, root = 0)
w_acc 		= comm.bcast(w_acc, root = 0)
quads_bcast = comm.bcast(quads_bcast, root = 0)
others 		= comm.bcast(others, root = 0)

subsys 		= Agent(N, ucoeff, delta, cs, M, mu, qvals, ncoloc, q_coeff)
subsys()
system 		= Principal(subsys)

results_all = [system.optimize_contract(others, restarts = jobs)]
results_all = comm.gather(results_all, root = 0)
if rank == 0:
	final_result = results_all[np.argmax([results_all[i][0]['se_obj'] for i in range(size)])]
	with open('result.pickle', 'wb') as myfile:
		pickle.dump((results_all, final_result), myfile)


	filename = "result.pickle"
	with open(filename, 'rb') as myfile:
			all_res = pickle.load(myfile)[0]
	final_res = all_res[np.argmax([all_res[i][0]['se_obj'] for i in range(size)])]
	print final_res

