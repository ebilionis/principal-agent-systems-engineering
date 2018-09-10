from theano import tensor as T
from theano import function
import theano
from theano.tensor.shared_randomstreams import RandomStreams
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from dpaspgp import *
from scipy.special import roots_hermitenorm
from theano.tensor.shared_randomstreams import RandomStreams
import time
from mpi4py import MPI
import pickle
import time
import scipy.optimize as opt
THEANO_FLAGS = 'optimizer=None'

def split(container, count):
    return [container[_i::count] for _i in range(count)]


class SubSys(object):
	def __init__(self, N, t_kappa, t_delta, t_cs, M, t_mu, t_qvals):
		self.N = N
		self.t_kappa = t_kappa
		self.t_delta = t_delta
		self.t_cs = t_cs
		self.M = M
		self.t_mu = t_mu
		self.transfer_list = []
		self.sse_exp_pi_list = []
		quads, weights, weight_acc = roots_hermitenorm(10000, mu=True)
		quads_bcast = np.array([quads]*M).T
		weights = np.array(weights)
		self.t_rvs_bcast = T.as_tensor(quads_bcast)
		self.t_rvs = T.as_tensor(quads)
		self.t_ws = T.as_tensor(weights)
		self.t_accweight = T.as_tensor(weight_acc)
		self.t_qvals = t_qvals
		self.dx = self.M * N


	def Q(self, t_eff, t_xi):

		def help_fun(ith, eff):
			return self.t_kappa[ith] * T.pow(eff[ith], 0.5) + self.t_delta[ith] * t_xi

		self.t_Q, upd_Q = theano.scan(fn = help_fun,
        	                          sequences=[T.arange(self.N, dtype='int32')],
            	                      non_sequences=[t_eff])
		return self.t_Q

	def sse_cost(self, t_eff):

		def help_fun(ith, eff):
			return self.t_cs[ith] * T.pow(eff[ith], 2)

		self.t_C, upd_C = theano.scan(fn = help_fun,
        	                          sequences=[T.arange(self.N, dtype='int32')],
            	                      non_sequences=[t_eff])
		return self.t_C

	def sse_transfer(self, t_parameters, t_eff):

		def help_fun(ith, param, eff):

  		 	return T.dot(self.t_ws, T.dot(param[ith*self.M:(ith+1)*self.M], \
						T.transpose(1.0 / (1.0 + T.exp(-100*(self.Q(eff, \
						self.t_rvs_bcast)[ith] - self.t_mu)))))) / self.t_accweight
  		self.t_sse_transfer, upd_transfer = theano.scan(fn = help_fun,
        	                          		sequences=[T.arange(self.N, dtype='int32')],
            	                      		non_sequences=[t_parameters, t_eff])
  		return self.t_sse_transfer

	def sse_utility(self, t_parameters, t_eff):

		def help_fun(ith, param, eff):
  			return  T.dot(self.t_ws, T.dot(param[ith*self.M:(ith+1)*self.M], \
						T.transpose(1.0 / (1.0 + T.exp(-100*(self.Q(eff, \
						self.t_rvs_bcast)[ith] - self.t_mu)))))) / self.t_accweight -\
  						self.sse_cost(eff)[ith]

		self.t_sse_utility, updates_sse = theano.scan(fn = help_fun,
	                       sequences=[T.arange(self.N, dtype='int32')],
	                       non_sequences = [t_parameters, t_eff])
		
		self.t_grad_sse_utility, upd = theano.scan(fn = lambda i, f, x:\
													T.grad(f[i], x)[i*self.M:(i+1)*self.M],
													sequences = [T.arange(self.N, dtype='int32')],
													non_sequences = [self.t_sse_utility, t_parameters])

		return self.t_sse_utility, self.t_grad_sse_utility


	def build_sse_effort(self, t_parameters, t_eff):
		opt_fun = self.sse_utility(t_parameters, t_eff)[0]
		t_r   = T.dvector('r')
		self.sse_opt_prob = []
		for i in range(self.N):
			self.sse_opt_prob += [ConstrainedOptimizationProblem(self.N, t_eff, t_r, 
												-self.t_sse_utility[i],
												t_xl = 0.0,
												t_xu = 1.0,
												t_other = [t_parameters])]

		# self.sse_opt_problem, upd =  theano.scan(fn = help_fun,
		# 										sequences = [T.arange(self.N, dtype='int32')],
		# 										non_sequences = [])
		return self.sse_opt_prob

	def solve_sse_effort(self, param):
		res_x = np.array(np.zeros(self.N))
		res_obj = np.array(np.zeros(self.N))
		#do multiple restarts 
		for ran in range(self.M + 1):
			lb = (ran + 0.0) / (self.M+1.0)
			ub = (ran + 1.0) / (self.M+1.0)
			x0 = np.random.uniform(lb, ub, 1)
			for i in range(self.N):
				p0 = np.zeros(self.N)
				p0[i] = x0
				res = self.sse_opt_prob[i].solve(p0, [], [param])
				if -res['obj'] > res_obj[i] and res['status'] == 0:
					res_obj[i] = -res['obj']
					res_x[i] = res['x'][i]
		return res_x, res_obj

	def subsys_exp_velue(self, t_eff):

		def help_fun(ith, eff):
			return T.dot(self.t_ws, 1.0 / (1.0 + T.exp(-200.*(self.Q(eff, self.t_rvs)[ith] \
				- self.t_qvals[ith])))) / self.t_accweight

		self.t_subsys_exp_value, updates_sys = theano.scan(fn = help_fun,
        	                                sequences=[T.arange(self.N, dtype='int32')],
            	                            non_sequences=[t_eff])
		self.t_sys_tot_value = T.sum(self.t_subsys_exp_value)
		return self.t_sys_tot_value

	def __call__(self, t_parameters, t_eff):

		self.sse_transfer(t_parameters, t_eff)
		self.sse_utility(t_parameters, t_eff)
		self.subsys_exp_velue(t_eff)
		self.sse_transfer_compile = function([t_parameters, t_eff], self.t_sse_transfer)
		self.sse_utility_compile = function([t_parameters, t_eff], self.t_sse_utility)
		self.sys_tot_value_compile = function([t_eff], self.t_sys_tot_value)
		self.build_sse_effort(t_parameters, t_eff)
		self.grad_sse_utility_compile = []
		for i in range(self.N):
			self.grad_sse_utility_compile += [function([t_parameters, t_eff], self.t_grad_sse_utility[i])]






if __name__ == '__main__':

	N = 1
	t_kappa = T.as_tensor(np.array([1.4]))
	t_delta = T.as_tensor(np.array([0.2]))
	t_cs = T.as_tensor(np.array([0.3]))
	M = 20
	t_mu = T.as_tensor(np.linspace(0.0, 1.3, M))
	t_qvals = T.as_tensor(np.array([1.0]))
	t_parameters = T.dvector('parameters')
	t_other = [t_parameters]
	t_r = T.dvector('r')
	t_eff = T.dvector('eff')

	quads, weights, weight_acc = roots_hermitenorm(10000, mu=True)
	quads_bcast = np.array([quads]*M).T
	weights = np.array(weights)
	t_rvs_bcast = T.as_tensor(quads_bcast)
	t_rvs = T.as_tensor(quads)


	sys = SubSys(N, t_kappa, t_delta, t_cs, M, t_mu, t_qvals)
	sys(t_parameters, t_eff)
	eff = np.linspace(0.0, 1.0, 100)

	h1 = []
	h2 = []
	h3 = []
	h4 = []
	Q = function([t_eff], sys.Q(t_eff, t_rvs))
	cost = function([t_eff], sys.sse_cost(t_eff))

	transfer = function([t_parameters, t_eff], sys.t_sse_transfer)
	utility = function([t_parameters, t_eff], sys.t_sse_utility)
	value = function([t_eff], sys.t_sys_tot_value)
	a = value(np.array([1,1,1]))
	print a

	for i in eff:
		h1 += [np.mean(Q(np.array([i])), axis=1)]
		h2 += [cost(np.array([i]))]
	fig, ax = plt.subplots()
	ax.plot(eff, h1)
	plt.ylabel('Q')
	plt.savefig('q.png')
	fig, ax = plt.subplots()
	ax.plot(eff, h2)
	plt.ylabel('c')
	plt.savefig('c.png', dpi = 300)
	fig3, ax3 = plt.subplots()
	fig4, ax4 = plt.subplots()
	for j in range(1):
		if j %100 == 0:
			print j
		param = np.random.uniform(0.0, 3.0, N*M)
		param = np.array([1.34365009e-17, 1.23781406e-16, 3.42458401e-17, 1.35878705e-17,
       7.55155663e-17, 2.59610622e-17, 4.98100823e-17, 2.41841910e-17,
       4.50151862e-17, 2.48295977e-17, 6.12169763e-17, 1.41423206e-16,
       2.21752411e-17, 4.10283458e-17, 1.42394255e-16, 1.44250296e-17,
       2.46226378e-17, 1.58282510e-17, 3.18774817e-01, 4.23425676e-02])
		param /= np.sum(param)
		h3 = []
		h4 = []
		for i in eff:

			h3 += [transfer(param, np.array([i]))]
			h4 += [sys.sse_utility_compile(param, np.array([i]))]

		ax3.plot(eff, h3)
		ax4.plot(eff, h4)
	ax3.set_ylabel('t')
	ax4.set_ylabel('u')
	fig3.savefig('t.png', dpi=300)
	fig4.savefig('u.png', dpi=300)
	t0 = time.time()
	param = np.array([1.34365009e-17, 1.23781406e-16, 3.42458401e-17, 1.35878705e-17,
       7.55155663e-17, 2.59610622e-17, 4.98100823e-17, 2.41841910e-17,
       4.50151862e-17, 2.48295977e-17, 6.12169763e-17, 1.41423206e-16,
       2.21752411e-17, 4.10283458e-17, 1.42394255e-16, 1.44250296e-17,
       2.46226378e-17, 1.58282510e-17, 3.18774817e-01, 4.23425676e-02])
	x,o = sys.solve_sse_effort(param)
	t1 = time.time()
	print 'time: ', t1 - t0
	print x
	print o
	print sys.grad_sse_utility_compile[1](param, x)
	plt.show()

	






