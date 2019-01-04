import pickle
import numpy as np
all_res = []
size = 20
for i in range(size):
	filename = 'proc'+str(i+1)+'.res'
	with open(filename, 'rb') as myfile:
		all_res.append(pickle.load(myfile))
final_res = all_res[np.argmax([all_res[i][0]['se_obj'] for i in range(size)])]

print final_res
