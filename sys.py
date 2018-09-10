from subsys import *
import scipy.optimize as opt
class Sys(object):
	def __init__(self, subsys):
		self.subsys = subsys
		t_parameters = T.dvector('parameters')
		t_eff = T.dvector('eff')
		if hasattr(self.subsys, 'sse_transfer_compile'):
			pass
		else:
			self.subsys(t_parameters, t_eff)

		self.sse_transfer = function([t_parameters, t_eff], subsys.t_sse_transfer)
		self.sse_utility = function([t_parameters, t_eff], subsys.t_sse_utility)
		self.sys_value = function([t_eff], subsys.t_sys_tot_value)
		self.sse_eff = np.array([0.]*self.subsys.N)
	def neg_se_obj_fun(self, param, eff=None):
		self.sse_eff, self.sse_obj = self.subsys.solve_sse_effort(param)

		tot_val = self.sys_value(self.sse_eff)
		tot_transfer = np.sum(self.sse_transfer(param, self.sse_eff))

		return tot_transfer - tot_val

	def t_grad_neg_fun(self, t_param, t_eff):

		self.subsys.sse_transfer(t_param, t_eff)
		self.subsys.subsys_exp_velue(t_eff)
		self.t_neg_tot = T.sum(self.subsys.t_sse_transfer) - self.subsys.t_sys_tot_value
		self.t_obj_jac = T.grad(self.t_neg_tot, [t_param])[0]
		return self.t_obj_jac

	def optimize_contract(self, restarts = 50):
		N = self.subsys.N
		M = self.subsys.M
		t_eff = T.dvector('eff')
		t_param = T.dvector('param')
		t_f_jac = self.t_grad_neg_fun(t_param, t_eff)

		nvar = self.subsys.M * self.subsys.N
		a0 = np.concatenate([np.random.uniform(-0.5, 1.4, 1), np.random.uniform(0.0, 1.4, nvar-1)])
		self.sse_eff, self.sse_obj = self.subsys.solve_sse_effort(a0)


		# The jacobian of the 
		f_jac = function([t_param, t_eff], t_f_jac)

		compare = np.infty
		result = None
		cons =  [{'type': 'ineq', 'fun': lambda x: -self.neg_se_obj_fun(x), 
		'jac':lambda x: -f_jac(x, self.sse_eff)}]
		cons += [{'type': 'ineq', 'fun': lambda x: self.sse_obj[i], \
				'jac': lambda x:self.subsys.grad_sse_utility_compile[i](x, self.sse_eff)} for i in range(N)] # this should be corercted
		#cons += [{'type': 'ineq', 'fun': lambda x: 1.0 - np.sum(x[i*M:(i+1)*M])} for i in range(N)]
		bnds = ([(-0.5, 1.5)]+[(0.0, 1.5)]*(nvar-1))
		for r in range(restarts):
			
			res = opt.minimize(self.neg_se_obj_fun, x0 = a0, args=(self.sse_eff,), method = 'slsqp', 
								jac=f_jac, constraints = cons, options={'disp':True},  bounds = bnds)

			a0 = np.concatenate([np.random.uniform(-0.5, 1.4, 1), np.random.uniform(0.0, 1.4, nvar-1)]) # change this later

			print r
			if res.fun < compare:
				compare = res.fun
				result = res
				print '********************************************************'
				print 'SE objective: ', -res.fun
				print 'sse_utility: ', self.sse_obj
				print 'sse_effort: ', self.sse_eff
				print 'sse_transfer: ', self.sse_transfer(res.x, self.sse_eff, )
				print 'parameters: ', res.x
				print '********************************************************'
				print result
		print result



if __name__ == '__main__':

	N = 1
	t_kappa = T.as_tensor(np.array([1.4]))
	t_delta = T.as_tensor(np.array([.2]))
	t_cs = T.as_tensor(np.array([0.3]))
	M = 6
	t_mu = T.as_tensor(np.linspace(0.0, 1.3, M))
	t_qvals = T.as_tensor(np.array([1.]))
	t_parameters = T.dvector('parameters')
	t_other = [t_parameters]
	t_r = T.dvector('r')
	t_eff = T.dvector('eff')

	subsys = SubSys(N, t_kappa, t_delta, t_cs, M, t_mu, t_qvals)
	sys = Sys(subsys)
	sys.optimize_contract(20)


