# incentives
Computation of Optimal Contracts for Systems Engineering

To run the code:

mpi -n "enter the number of processors" python run.py


To set the agent types, in run.py:

1) number_opt: This parameter is the number of restarts for the outer optimization problem

2) N         : Number of agents

3) q_coeff   : List of the parameters to set the quality function

4) ucoeff    : The coefficient to set the risk behavior of the agent

5) delta     : The uncertainty in the quality

6) cs        : The cost coefficient of the agent