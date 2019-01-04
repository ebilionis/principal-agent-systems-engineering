import sys  
sys.path.append('/home/ssafarkh/dpasgp-master/ieee')
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
import pickle
import time
import scipy.optimize as opt

def split(container, count):
    return [container[_i::count] for _i in range(count)]


class Agent(object):
	def __init__(self, N, ucoeff, delta, cs, M, mu, qvals, ncoloc):
		self.N = N
		self.ucoeff = ucoeff
		self.delta = delta
		self.cs = cs
		self.M = M

		self.transfer_list = []
		self.sse_exp_pi_list = []
		quads, weights, weight_acc = roots_hermitenorm(10000, mu=True)
		quads_bcast = np.array([quads]*M).T
		weights = np.array(weights)
		self.t_rvs_bcast = T.as_tensor(quads_bcast)
		self.t_rvs = T.as_tensor(quads)
		self.t_ws = T.as_tensor(weights)
		self.t_accweight = T.as_tensor(weight_acc)
		self.qvals = qvals
		self.ncoloc = ncoloc

		self.dx = self.M * N
		self.t_ei1 = T.dvector('ei1')
		self.t_ei2 = T.dvector('ei2')
		self.t_e = T.dvector('e')
		self.t_ai = T.dvector('ai')
		self.t_xi_i = T.dmatrix('xi_i')

		self.t_w = T.dmatrix('w')
		self.t_w_acc = T.dscalar('w_acc')
		self.t_Qi1 = T.dmatrix('Qi1')
		self.t_mu = theano.shared(mu.flatten().reshape(1, -1), broadcastable = (True, False))
		self.ell = -20.0

		self.t_parameters = T.dvector('parameters')




	def build_variables(self):
		self.t_c0 = self.cs * self.t_ei1[0]**2
		self.t_Q0 = 0.64145697 * T.log(self.t_ei2[0]+0.04650573)+1.97937449 + self.delta * self.t_xi_i
		self.t_grad_Qe0 = T.grad(T.mean(self.t_Q0), self.t_ei2)
		
#		self.t_t0 = T.dot(self.t_ai, T.transpose(1.0 / (1.0 + T.exp(self.ell * (self.t_Qi1 - self.t_mu)))))
		self.t_t0 = self.t_ai[0] + self.t_ai[1]*(1.0 / (1.0 + T.exp(self.ell * (self.t_Qi1 - self.t_ai[2])))) + \
                            self.t_ai[3] * (self.t_Qi1-self.t_ai[2])*(1.0 / (1.0 + T.exp(self.ell * (self.t_Qi1 - self.t_ai[2]))))
		self.t_t0_exp = T.dot(self.t_w, self.t_t0) / self.t_w_acc

		self.t_sysval0 = 1.0 / (1.0 + T.exp(-50*(self.t_Qi1 - 1.0)))
		self.t_sysval0_exp = T.dot(self.t_w, self.t_sysval0) / self.t_w_acc

		self.t_sse_pi0 = self.t_t0 - self.t_c0
		self.t_sse_pi0_exp = self.t_t0_exp - self.t_c0

		# utility of sse
		self.a = 1.0 / (1.0 - T.exp(-self.ucoeff))
		self.b = self.a
		self.t_sse_util0 = T.flatten(self.a - self.b * T.exp(-self.ucoeff * self.t_sse_pi0))
		self.t_sse_util0_exp = T.flatten(self.a - self.b * T.exp(-self.ucoeff * self.t_sse_pi0_exp))
#		self.t_sse_util0 = T.flatten(self.t_sse_pi0)
#		self.t_sse_util0_exp = T.flatten(self.t_sse_pi0_exp)
		
		# derivatives

		self.t_grad_tQ0 = T.grad(self.t_t0[0,0], self.t_Qi1)
		self.t_grad_tQ0_exp = T.dot(self.t_w, self.t_grad_tQ0) / self.t_w_acc

		self.t_grad_ta0 = T.grad(self.t_t0[0,0], self.t_ai)
		self.t_grad_ta0_exp = T.dot(self.t_w, self.t_grad_ta0) / self.t_w_acc

		self.t_grad_VQ0 = T.grad(self.t_sysval0[0,0], self.t_Qi1)
		self.t_gard_VQ0_exp = T.dot(self.t_w, self.t_grad_VQ0) / self.t_w_acc


		


	def cost(self):

		self.t_cost = theano.clone(self.t_c0, replace = {self.t_ei1: self.t_e}, strict = False)

	def Q(self):

		self.t_Q = theano.clone(self.t_Q0, replace = {self.t_ei2: self.t_e}, strict = False)

	def grad_Q_e(self):
		
		self.t_grad_Q_e = theano.clone(self.t_grad_Qe0, replace = {self.t_Q0:self.t_Q, self.t_ei2:self.t_e}, strict = False)

	def transfer(self):
		
		self.t_tr = theano.clone(self.t_t0, replace = {self.t_Qi1: self.t_Q}, strict = False)

	def exp_transfer(self):
		
		self.t_tr_exp = theano.clone(self.t_t0_exp, replace = {self.t_t0: self.t_tr}, strict = False)

	def system_value(self):

		self.t_sysval = theano.clone(self.t_sysval0, replace = {self.t_Qi1: self.t_Q}, strict = False)

	def exp_system_value(self):

		self.t_sysval_exp = theano.clone(self.t_sysval0_exp, replace = {self.t_sysval0: self.t_sysval}, strict = False)

	def grad_transfer_quality(self):
		
		self.t_grad_trQ = theano.clone(self.t_grad_tQ0, replace = {self.t_t0: self.t_tr, self.t_Qi1: self.t_Q}, strict = False)

	def exp_grad_transfer_quality(self):
		
		self.t_grad_trQ_exp = theano.clone(self.t_grad_tQ0_exp, replace = {self.t_grad_tQ0: self.t_grad_trQ}, strict = False)

	def grad_transfer_a(self):

		self.t_grad_tra = theano.clone(self.t_grad_ta0, replace = {self.t_t0: self.t_tr, self.t_Qi1:self.t_Q}, strict = False)

	def exp_grad_transfer_a(self):

#		self.t_grad_tra_exp = T.dot(self.t_w, self.t_grad_tra)
		self.t_grad_tra_exp = theano.clone(self.t_grad_ta0_exp, replace= {self.t_grad_ta0: self.t_grad_tra}, strict = False)

	def grad_sysval_Q(self):

		self.t_grad_VQ = theano.clone(self.t_grad_VQ0, replace = {self.t_Qi1: self.t_Q}, strict = False)

	def exp_grad_sysval_Q(self):

#		self.t_grad_VQ_exp = T.dot(self.t_w, self.t_grad_VQ)
		self.t_grad_VQ_exp = theano.clone(self.t_gard_VQ0_exp, replace = {self.t_sysval0:self.t_sysval}, strict = False)

	def sse_pi(self):

		self.t_sse_pi  = theano.clone(self.t_sse_pi0, replace = {self.t_t0: self.t_tr, self.t_c0: self.t_cost}, strict = False)

	def exp_sse_pi(self):

		self.t_sse_pi_exp = theano.clone(self.t_sse_pi0_exp, replace = {self.t_t0_exp:self.t_tr_exp, self.t_c0: self.t_cost}, strict = False)

	def sse_utility(self):

		self.t_sse_util = theano.clone(self.t_sse_util0, replace = {self.t_sse_pi0: self.t_sse_pi}, strict = False)
#		self.t_sse_util = self.t_tr - self.t_cost
	def exp_sse_utility(self):

		self.t_sse_util_exp = theano.clone(self.t_sse_util0_exp, replace = {self.t_sse_pi0_exp: self.t_sse_pi_exp}, strict = False)
#		self.t_sse_util_exp  = self.t_tr_exp - self.t_cost

	def build_sse_opt_problem(self):

		t_g = [self.t_sse_util_exp[0]]
		self.sse_opt_prob = ConstrainedOptimizationProblem(self.N, self.t_e, self.t_ai, 
												-self.t_sse_util_exp[0],
												t_g,
												t_gl = 0.0,
												t_gu = POSITIVE_INFINITY,
												t_xl = 0.0,
												t_xu = 1.0,
												t_other = [self.t_w, self.t_xi_i, self.t_w_acc])
		
		return self.sse_opt_prob


	def solve_sse_opt_prob(self, param, oth):
		res_x          = np.array(np.zeros(self.N))
		res_obj        = np.array(np.zeros(self.N))
		res_grad_p_obj = np.array(np.zeros(self.M))
		res_grad_p_x   = np.array(np.zeros(self.M))
		#do multiple restarts 

		for ran in range(40):
			lb = (ran + 0.0) / 40.0
			ub = (ran + 1.0) / 40.0
			x0 = np.random.uniform(lb, ub, 1)
			for i in range(self.N):
				p0 = np.zeros(self.N)
				p0[i] = x0
				res = self.sse_opt_prob.solve(p0, param, other = oth, get_grad = True)
				if -res['obj'] > res_obj[i] and res['status'] == 0 and not np.any(np.isnan(res['grad_p_x'])):
					res_obj[i]       = -res['obj']
					res_x[i]         =  res['x'][i]
					res_grad_p_obj   =  res['grad_p_obj']
					res_grad_p_x     =  res['grad_p_x']
		return res_x, res_obj, res_grad_p_obj, res_grad_p_x





	def __call__(self):
		
		self.build_variables()
		self.Q()
		self.cost()
		self.transfer()
		self.exp_transfer()
		self.system_value()
		self.exp_system_value()
		self.sse_pi()
		self.exp_sse_pi()
		self.sse_utility()
		self.exp_sse_utility()
		self.build_sse_opt_problem()

		self.grad_Q_e()
		self.grad_transfer_quality()
		self.exp_grad_transfer_quality()

		self.grad_transfer_a()
		self.exp_grad_transfer_a()

		self.grad_sysval_Q()
		self.exp_grad_sysval_Q()



		self.Q_compiled            = function([self.t_e, self.t_xi_i], self.t_Q)
		self.cost_compiled         = function([self.t_e], self.t_cost)
		self.tr_compiled           = function([self.t_e, self.t_ai, self.t_xi_i], self.t_tr)
		self.tr_exp_compiled       = function([self.t_e, self.t_ai, self.t_w, self.t_xi_i, self.t_w_acc], self.t_tr_exp)
		self.sysval_compiled       = function([self.t_e, self.t_xi_i], self.t_sysval)
		self.sysval_exp_compiled   = function([self.t_e, self.t_w, self.t_xi_i, self.t_w_acc], self.t_sysval_exp)
		self.sse_util_compiled     = function([self.t_e, self.t_ai, self.t_xi_i], self.t_sse_util)
		self.sse_util_exp_compiled = function([self.t_e, self.t_ai, self.t_w, self.t_xi_i, self.t_w_acc], self.t_sse_util_exp)

		self.grad_Qe_compiled      = function([self.t_e, self.t_xi_i], self.t_grad_Q_e)

		self.grad_trQ_compiled     = function([self.t_e, self.t_ai, self.t_xi_i], self.t_grad_trQ)

		self.grad_tra_compiled     = function([self.t_e, self.t_ai, self.t_xi_i], self.t_grad_tra)

		self.grad_VQ_compiled      = function([self.t_e, self.t_xi_i], self.t_grad_VQ)





if __name__ == '__main__':

	N = 1

	kappa = np.array([1.3])

	delta = np.array([0.2])

	cs = np.array([0.3])

	M = 6

	ncoloc = 1000

	mu = np.linspace(0.5, 1.3, M)

	qvals = np.array([1.0])

	sys = Agent(N, kappa, delta, cs, M, mu, qvals, ncoloc)

	quads, weights, w_acc = roots_hermitenorm(ncoloc, mu=True)

	quads_bcast = np.array([quads]*M).T

	weights = np.array(weights)

	sys()
	param1 = np.array([1.00000606e-06, 1.02164096e-02, 6.95899824e-04, 1.00000072e-06,
       1.25377477e-01, 3.85058853e-01])
	# test the functions

	# print sys.tr_compiled(np.array([0.3]), param, np.array([[0.1]*M]))
	
	# print sys.tr_exp_compiled(np.array([1.]), param, weights.reshape(1, -1) , quads_bcast, w_acc)

	# print sys.sysval_compiled(np.array([.6]), np.array([[1.]]))

	# print sys.sysval_exp_compiled(np.array([0.9]), weights.reshape(1,-1), quads.reshape(-1, 1), w_acc)

	# print sys.sse_util_compiled(np.array([0.3]), np.array([1.0]*M), np.array([[0.1]*M]))

	# print sys.sse_util_exp_compiled(np.array([.3]), np.array([.1]*M), weights.reshape(1, -1), quads_bcast, w_acc)
	
	h1 = []
	e = np.linspace(0,1,500)

	for i in e:
		h1 += [sys.sse_util_exp_compiled(np.array([i]), param1, weights.reshape(1, -1), quads_bcast, w_acc)]
	plt.plot(e, h1)
	plt.xlabel('e')
	plt.ylabel('$u_{sSE}$')
	plt.legend()
	plt.savefig('1.png', dpi=300)
	# plt.show()
	

	# a, b, c, d = sys.solve_sse_opt_prob(param, [weights.reshape(1, -1), quads_bcast, w_acc])
	# print a
	# print b
	# print c
	# print d

	# print sys.grad_trQ_compiled(np.array([0.3]), np.array([1.0]*M), np.array([[quads[40]]*M]))
	# sum =  0.0
	# t0 = time.time()
	# ttt = np.ndarray(shape = (ncoloc, 1))
	# for i in range(ncoloc):
	# 	ttt[i,:] = sys.grad_trQ_compiled(np.array([0.7]), np.array([1.0]*M), np.array([[quads[i]]*M]))[0,0]
	# print ttt
	# print 'trQ', np.dot(weights, ttt)/w_acc

	# ttt = np.ndarray(shape = (ncoloc, M))
	# for i in range(ncoloc):
	# 	ttt[i,:] = sys.grad_tra_compiled(np.array([0.7]), np.array([1.0]*M), np.array([quads[i]]*M).reshape(1,-1))
	# print ttt
	# print 'tra', np.dot(weights, ttt) / w_acc


	# ttt = np.ndarray(shape=(ncoloc, 1))

	# for i in range(ncoloc):
	# 	 ttt[i, :] = sys.grad_VQ_compiled(np.array([.7]), np.array([[quads[i]]]))[0][0]
	# print ttt
	# print 'VQ', np.dot(weights, ttt) / w_acc
	
	# t1 = time.time()
	# print 't', t1 - t0
	plt.show()












