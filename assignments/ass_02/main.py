import numpy as np
import os
import scipy.io as io

# import data
filepath = os.path.abspath('./../../../data/ass_02/ct_data.mat')
assert os.path.exists(filepath), 'Please download the dataset. I cannot find it at: %s' %(filepath)
data = io.loadmat(filepath)

X_train = data['X_train']
X_val = data['X_val']
X_test = data['X_test']
