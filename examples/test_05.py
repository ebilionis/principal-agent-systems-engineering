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

agent_type11 = AgentType(LinearQualityFunction(1.5, 0.2),
                        QuadraticCostFunction(0.1),
                        ExponentialUtilityFunction())

agents = Agent([agent_type11])

t = RequirementPlusIncentiveTransferFunction(gamma=30.)

p = PrincipalProblem(ExponentialUtilityFunction(),
                    RequirementValueFunction(1, gamma=50.),
                    agents, t)
p.compile()

a3 = np.linspace(0,2.,1000)
y = []
for i in a3:
    print i
    a = [0.01186108, 0.08077786] + [i] + [0.23217116]
    y.append(p.evaluate(a)['exp_u_pi_0'])
plt.plot(a3, y)
plt.xlabel('a2')
plt.ylabel('$u_0$')
plt.show()