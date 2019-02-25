import pickle
import sys
with open(sys.argv[1], 'r') as f:
    res = pickle.load(f)
    print(res)
