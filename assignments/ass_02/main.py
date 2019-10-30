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

y_train = data['y_train']
y_val = data['y_val']
y_test = data['y_test']


# question 1a
def mean_with_sterror(x):
    m = x.mean()
    sigma = x.std(ddof = 1)
    sterror = sigma / np.sqrt(x.shape[0])
    return m, sterror



m, err = mean_with_sterror(y_val)
print("Mean estimation of y_val is : %.3f with standard error: %.3f" %(m, err))

N2 = 5785
m, err = mean_with_sterror(y_train[N1:N2])
print("Mean estimation of first %d values of y_train is : %.3f with standard error: %.3f" %(N, m, err))


list_m = []
list_stderr = []
np.random.seed(2)
N = 500
for i in range(100):
    np.random.shuffle(y_train)
    m, err = mean_with_sterror(y_train[:N])
    list_m.append(m)
    list_stderr.append(err)

np.mean(list_m)
np.std(list_m)
