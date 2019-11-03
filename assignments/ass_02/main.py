import numpy as np
import os
import copy
import scipy.io as io
from ct_support_code import *

# import data
filepath = os.path.abspath('./../../../data/ass_02/ct_data.mat')
assert os.path.exists(filepath), 'Please download the dataset. I cannot find it at: %s' %(filepath)
data = io.loadmat(filepath)

X_train = data['X_train']
X_val = data['X_val']
X_test = data['X_test']

y_train = data['y_train']
y_val = data['y_val']
y_test = data['y_test']

N_train = X_train.shape[0]
N_val = X_val.shape[0]
N_test = X_test.shape[0]


############################### Question 1 ###################################
# question 1a
def mean_with_sterror(x):
    m = x.mean()
    sigma = x.std(ddof = 1)
    sterror = sigma / np.sqrt(x.shape[0])
    return m, sterror


m, err = mean_with_sterror(y_val)
print("Mean estimation of y_val is : %.3f with standard error: %.3f" %(m, err))

N = 5785
m, err = mean_with_sterror(y_train[:N])
print("Mean estimation of first %d values of y_train is : %.3f with standard error: %.3f" %(N, m, err))

def small_experiment():
    list_m = []
    list_stderr = []
    np.random.seed(2)
    N = 500
    y_train_tmp = copy.deepcopy(y_train)
    for i in range(1000):
        np.random.shuffle(y_train_tmp)
        m, err = mean_with_sterror(y_train_tmp[:N])
        list_m.append(m)
        list_stderr.append(err)

    print("Mean estimation of y_train from %d iid samples, in 1000 different executions has mean: %.3f and standard deviation: %.3f" %(N, np.mean(list_m), np.std(list_m, ddof=1)))

    print("Standard error estimation of y_train from %d iid samples, in 1000 different executions has mean: %.3f and standard deviation: %.3f" %(N, np.mean(list_stderr), np.std(list_stderr, ddof = 1)))


# question 1b
threshold = 10e-10

# find indices of constant features 
ind_const_features = np.where(X_train.var(0) <= threshold)[0]

# find_indices of duplicate features
duplicates = []
for j in range(X_train.shape[1]):
    f1 = X_train[:,j]
    tmp = X_train[:, j+1:] - np.expand_dims(f1, -1)
    indices = np.where((np.var(tmp, 0) <= threshold))[0] + j + 1
    duplicates.append(indices)

duplicates_dict = {}    
for i, val in enumerate(duplicates):
    if len(val) > 0:
        duplicates_dict[i] = val
    
ind_duplicate_features = np.concatenate(duplicates).ravel()
ind_duplicate_features = np.sort(np.unique(ind_duplicate_features))


# merge indices
ind_excluded_features = np.concatenate((ind_const_features, ind_duplicate_features)).ravel()
ind_excluded_features = np.sort(np.unique(ind_excluded_features))

# redefine train, val, test sets
X_train = np.delete(X_train, ind_excluded_features, axis = 1)
X_val = np.delete(X_val, ind_excluded_features, axis = 1)
X_test = np.delete(X_test, ind_excluded_features, axis = 1)
D = X_train.shape[1]

# Question 2

# data augmentation
alpha = 10
reg = np.sqrt(alpha) * np.eye(D, D)
X1 = np.concatenate( (X_train, np.ones((N_train, 1)) ), axis = 1)
reg1 = np.concatenate( (reg, np.zeros((reg.shape[1], 1)) ), axis = 1)
X_aug = np.concatenate( (X1, reg1), axis=0)
y_aug = np.concatenate( (y_train, np.zeros((D, 1))), axis = 0)

# lstsq
W, SSE, rank, singulars = np.linalg.lstsq(X_aug, y_aug, rcond=None)
W_lstsq = W[:-1]
b_lstsq = W[-1]

# gradient method
W_grad, b_grad = fit_linreg_gradopt(X_train, y_train, alpha)

