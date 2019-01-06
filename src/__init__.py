from theano import tensor as T
from theano import function
import theano
from theano.tensor.shared_randomstreams import RandomStreams
import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
import numpy as np
from dpaspgp import *
from scipy.special import roots_hermitenorm
from theano.tensor.shared_randomstreams import RandomStreams
import time
import pickle
import time
import scipy.optimize as opt
from .principal import *
from .agent import *