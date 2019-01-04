import pickle
import numpy as np
filename = "result.pickle"
with open(filename, 'rb') as myfile:
		all_res = pickle.load(myfile)[0]
# print all_res[0][0][2][0]['se_obj']
final_res = all_res[np.argmax([all_res[i][0]['se_obj'] for i in range(4)])]

print final_res

# print all_res
