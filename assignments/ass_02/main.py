import numpy as np
import os
import copy
import scipy.io as io
import matplotlib.pyplot as plt
from ct_support_code import *
plt.style.use('ggplot')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# import data
_filepath = os.path.abspath('./../../../data/ass_02/ct_data.mat')
assert os.path.exists(_filepath), 'Please download the dataset. I cannot find it at: %s' % (_filepath)
_data = io.loadmat(_filepath)

X_train = _data['X_train']
X_val = _data['X_val']
X_test = _data['X_test']

y_train = _data['y_train']
y_val = _data['y_val']
y_test = _data['y_test']

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

# ############################ Question 2 ##################################
# def fit_linreg(X, yy, alpha):
#     # data augmentation
#     D = X.shape[1]
#     N = X.shape[0]
#
#     reg = np.sqrt(alpha) * np.eye(D, D)
#     X1 = np.concatenate( (X, np.ones((N, 1)) ), axis = 1)
#     reg1 = np.concatenate( (reg, np.zeros((reg.shape[1], 1)) ), axis = 1)
#     X_aug = np.concatenate( (X1, reg1), axis=0)
#     y_aug = np.concatenate( (yy, np.zeros((D, 1))), axis = 0)
#
#     # lstsq
#     W, SSE, rank, singulars = np.linalg.lstsq(X_aug, y_aug, rcond=None)
#     W_lstsq = W[:-1]
#     b_lstsq = W[-1]
#     return W_lstsq, b_lstsq
#
# # least square method
# W_lstsq, b_lstsq = fit_linreg(X_train, y_train, 10)
#
# # gradient method
# alpha = 10
# W_grad, b_grad = fit_linreg_gradopt(X_train, np.squeeze(y_train), alpha)
#
# # Errors
# def compute_RMSE(X, y, w, b):
#     # expand_dims to all single dimensional arrays
#     if len(y.shape) == 1:
#         y = np.expand_dims(y, -1)
#
#     if len(w.shape) == 1:
#         w = np.expand_dims(w, -1)
#
#     # compute RMSE
#     y_bar = np.dot(X, w) + b
#     square_erros = np.square(y_bar - y)
#     RMSE = np.sqrt(np.mean(square_erros))
#     return RMSE
#
# RMSE_lstsq_tr = compute_RMSE(X_train, y_train, W_lstsq, b_lstsq)
# RMSE_lstsq_val = compute_RMSE(X_val, y_val, W_lstsq, b_lstsq)
# RMSE_lstsq_test = compute_RMSE(X_test, y_test, W_lstsq, b_lstsq)
#
# RMSE_grad_tr = compute_RMSE(X_train, y_train, W_grad, b_grad)
# RMSE_grad_val = compute_RMSE(X_val, y_val, W_grad, b_grad)
# RMSE_grad_test = compute_RMSE(X_test, y_test, W_grad, b_grad)
#
#
# ############################ Question 3 ##################################
# # Question 3i
# def fit_and_measure_on_projection(K):
#     alpha = 10
#     proj_mat = random_proj(D, K)
#
#     # projected X
#     X_train_proj = np.dot(X_train, proj_mat)
#     X_val_proj = np.dot(X_val, proj_mat)
#     X_test_proj = np.dot(X_test, proj_mat)
#
#     results = {"K": K}
#
#     # fitting
#     W_lstsq_proj, b_lstsq_proj = fit_linreg(X_train_proj, y_train, alpha)
#     W_grad_proj, b_grad_proj = fit_linreg_gradopt(X_train_proj, np.squeeze(y_train), alpha)
#
#     # RMSE
#     results['RMSE_lstsq_tr'] = compute_RMSE(X_train_proj, y_train, W_lstsq_proj, b_lstsq_proj)
#     results['RMSE_lstsq_val'] = compute_RMSE(X_val_proj, y_val, W_lstsq_proj, b_lstsq_proj)
#     results['RMSE_lstsq_test'] = compute_RMSE(X_test_proj, y_test, W_lstsq_proj, b_lstsq_proj)
#
#     results['RMSE_grad_tr'] = compute_RMSE(X_train_proj, y_train, W_grad_proj, b_grad_proj)
#     results['RMSE_grad_val'] = compute_RMSE(X_val_proj, y_val, W_grad_proj, b_grad_proj)
#     results['RMSE_grad_test'] = compute_RMSE(X_test_proj, y_test, W_grad_proj, b_grad_proj)
#
#     return results
#
# K = 10
# q3a_results_proj_10 = fit_and_measure_on_projection(K)
#
# K = 100
# q3a_results_proj_100 = fit_and_measure_on_projection(K)
#
#
# # Question 3ii
# _save_filename_png = os.path.abspath("./presentation/presentation_figures/fig_01.pdf")
# plt.figure()
# plt.title("Histogram of feature 45")
# plt.hist(X_train[45].ravel(), bins=30)
# plt.xlabel('value')
# plt.ylabel('number of samples')
# plt.savefig(_save_filename_png)
# plt.show()
#
# q3b_pcg = np.sum(np.logical_or(X_train == 0, X_train < 0)) / X_train.size
#
# def fit_and_measure_added_binaries():
#     alpha = 10
#
#     # projected X
#     def aug_fn(X): return np.concatenate([X, X == 0, X < 0], axis=1)
#
#     X_train_aug = aug_fn(X_train)
#     X_val_aug = aug_fn(X_val)
#     X_test_aug = aug_fn(X_test)
#
#     # fitting
#     W_lstsq, b_lstsq = fit_linreg(X_train_aug, y_train, alpha)
#     # W_grad_proj, b_grad_proj = fit_linreg_gradopt(X_train_proj, np.squeeze(y_train), alpha)
#
#     # RMSE
#     results = {}
#     results['RMSE_lstsq_tr'] = compute_RMSE(X_train_aug, y_train, W_lstsq, b_lstsq)
#     results['RMSE_lstsq_val'] = compute_RMSE(X_val_aug, y_val, W_lstsq, b_lstsq)
#     results['RMSE_lstsq_test'] = compute_RMSE(X_test_aug, y_test, W_lstsq, b_lstsq)
#
#     # results['RMSE_grad_tr'] = compute_RMSE(X_train_proj, y_train, W_grad_proj, b_grad_proj)
#     # results['RMSE_grad_val'] = compute_RMSE(X_val_proj, y_val, W_grad_proj, b_grad_proj)
#     # results['RMSE_grad_test'] = compute_RMSE(X_test_proj, y_test, W_grad_proj, b_grad_proj)
#
#     return results
#
# q3b_results_added_binaries = fit_and_measure_added_binaries()
#
#
# ############################ Question 4 ##################################
# # fit each class
# K = 10 # number of thresholded classification problems to fit
# mx = np.max(y_train)
# mn = np.min(y_train)
# hh = (mx-mn)/(K+1)
# thresholds = np.linspace(mn+hh, mx-hh, num=K, endpoint=True)
#
# alpha = 10
# weight_dict = {}
# for kk in range(K):
#     labels = y_train > thresholds[kk]
#
#     # fit logistic regression
#     ww, bb = fit_linreg_gradopt(X_train, np.squeeze(labels), alpha)
#     weight_dict[kk] = {}
#     weight_dict[kk]['w'] = ww
#     weight_dict[kk]['b'] = bb
#
# # create X_smart_proj
# weights = []
# bias = []
# for key, value in weight_dict.items():
#     weights.append(value["w"])
#     bias.append(value["b"])
# ww = np.stack(weights, axis=1)
# bb = np.expand_dims(np.stack(bias), 0)
#
# def sigmoid(x): return 1/ (1 + np.exp(-x))
# X_train_smart = sigmoid(np.dot(X_train, ww) + bb)
# X_val_smart = sigmoid(np.dot(X_val, ww) + bb)
# X_test_smart = sigmoid(np.dot(X_test, ww) + bb)
#
# # X_train_smart_1 = np.dot(X_train, ww) + bb
# # X_val_smart_1 = np.dot(X_val, ww) + bb
# # X_test_smart_1 = np.dot(X_test, ww) + bb
#
# W_smart, b_smart = fit_linreg(X_train_smart, y_train, alpha)
#
# q4_RMSE_smart_tr = compute_RMSE(X_train_smart, y_train, W_smart, b_smart)
# q4_RMSE_smart_val = compute_RMSE(X_val_smart, y_val, W_smart, b_smart)
# q4_RMSE_smart_test = compute_RMSE(X_test_smart, y_test, W_smart, b_smart)


############################ Question 5 ##################################
ww1, bb1, ww2, bb2 = fit_cnn_gradopt(X_train, np.squeeze(y_train), alpha=10)