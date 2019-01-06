import sys  
sys.path.append('../')
from src import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
N = 1
ucoeff = 0.0
delta = np.array([0.])
cs = np.array([0.3])
M = 4
mu = np.linspace(0.5, 1.3, M)
qvals = np.array([1.0])
ncoloc = 1000
q_coeff = [0., 1.0]
ag1 = Agent(N, ucoeff, delta, cs, M, mu, qvals, ncoloc, q_coeff)
quads, weights, w_acc = roots_hermitenorm(ncoloc, mu=True)
quads_bcast = np.array([quads]*M).T
weights = np.array(weights)
ag1()
param1 = np.array([0.0, 0.5, 1.0, 0.1])
param2 = np.array([0.1, 0.2, 1.1, 0.4])
param3 = np.array([0.0, 0.4, 0.8, 0.3])
# test the functions
e = np.linspace(0, 2.0, 1000)
y1 = []
y2 = []
y3 = []
for i in e:
	y1 += [ag1.tr_exp_compiled(np.array([i]), param1, weights.reshape(1, -1) , quads.reshape(-1, 1), w_acc)[0][0]]
	y2 += [ag1.tr_exp_compiled(np.array([i]), param2, weights.reshape(1, -1) , quads.reshape(-1, 1), w_acc)[0][0]]
	y3 += [ag1.tr_exp_compiled(np.array([i]), param3, weights.reshape(1, -1) , quads.reshape(-1, 1), w_acc)[0][0]]
plt.plot(e, y1, label='[0.0, 0.5, 1.0, 0.1]')
plt.plot(e, y2, label='[0.1, 0.2, 1.1, 0.4]')
plt.plot(e, y3, label='[0.0, 0.4, 0.8, 0.3]')
plt.legend()
plt.xlabel('q')
plt.ylabel('t')
plt.savefig('heaviside.png', dpi = 300)
y1 = []
y2 = []
y3 = []
fig, ax = plt.subplots()
for i in e:
	y1 += [ag1.grad_trQ_compiled(np.array([i]), param1, quads.reshape(-1, 1))[0]]
	y2 += [ag1.grad_trQ_compiled(np.array([i]), param2, quads.reshape(-1, 1))[0]]
	y3 += [ag1.grad_trQ_compiled(np.array([i]), param3, quads.reshape(-1, 1))[0]]
ax.plot(e, y1, label='[0.0, 0.5, 1.0, 0.1]')
ax.plot(e, y2, label='[0.1, 0.2, 1.1, 0.4]')
ax.plot(e, y3, label='[0.0, 0.4, 0.8, 0.3]')
ax.set_xlabel('q')
ax.set_ylabel('t`')
ax.legend()
fig.savefig('heaviside_derivative.png', dpi=300)


# print sys.tr_exp_compiled(np.array([1.]), param, weights.reshape(1, -1) , quads_bcast, w_acc)

# print sys.sysval_compiled(np.array([.6]), np.array([[1.]]))

# print sys.sysval_exp_compiled(np.array([0.9]), weights.reshape(1,-1), quads.reshape(-1, 1), w_acc)

# print sys.sse_util_compiled(np.array([0.3]), np.array([1.0]*M), np.array([[0.1]*M]))

# print sys.sse_util_exp_compiled(np.array([.3]), np.array([.1]*M), weights.reshape(1, -1), quads_bcast, w_acc)

# h1 = []
# e = np.linspace(0,1,500)

# for i in e:
# 	h1 += [sys.sse_util_exp_compiled(np.array([i]), param1, weights.reshape(1, -1), quads_bcast, w_acc)]
# plt.plot(e, h1)
# plt.xlabel('e')
# plt.ylabel('$u_{sSE}$')
# plt.legend()
# plt.savefig('1.png', dpi=300)
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
# plt.show()