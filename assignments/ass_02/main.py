import os
import copy
import scipy.io as io
import matplotlib.pyplot as plt
from ct_support_code import *
import copy
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
from ct_support_code import *

np.random.seed(1)
plt.style.use('ggplot')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# import data
_filepath = os.path.abspath('./../../../data/ass_02/ct_data.mat')
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
q1a1 = np.mean(y_train)
q1a2 = np.mean(y_val)


def mean_with_sterror(x):
    m = x.mean()
    sigma = x.std(ddof=1)
    sterror = sigma / np.sqrt(x.shape[0])
    return m, sterror

q1a3_m, q1a3_err = mean_with_sterror(y_val)
N = 5785
q1a4_m, q1a4_err = mean_with_sterror(y_train[:N])

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
    print("Mean estimation of y_train from %d iid samples, in 1000 different executions has mean: %.3f and standard deviation: %.3f" % (
        N, np.mean(list_m), np.std(list_m, ddof=1)))
    print(
        "Standard error estimation of y_train from %d iid samples, in 1000 different executions has mean: %.3f and standard deviation: %.3f" % (
        N, np.mean(list_stderr), np.std(list_stderr, ddof=1)))


# question 1b
_threshold = 10e-10

# find indices of constant features
q1b_ind_const_features = np.where(X_train.var(0) <= _threshold)[0]

# find_indices of duplicate features
_duplicates = []
for j in range(X_train.shape[1]):
    f1 = X_train[:, j]
    tmp = X_train[:, j + 1:] - np.expand_dims(f1, -1)
    indices = np.where((np.var(tmp, 0) <= _threshold))[0] + j + 1
    _duplicates.append(indices)

q1b_duplicates_dict = {}
for i, val in enumerate(_duplicates):
    if len(val) > 0:
        q1b_duplicates_dict[i] = val

q1b_ind_duplicate_features = np.concatenate(_duplicates).ravel()
q1b_ind_duplicate_features = np.sort(np.unique(q1b_ind_duplicate_features))

# merge indices
q1b_ind_excluded_features = np.concatenate((q1b_ind_const_features, q1b_ind_duplicate_features)).ravel()
q1b_ind_excluded_features = np.sort(np.unique(q1b_ind_excluded_features))

# redefine train, val, test sets
X_train = np.delete(X_train, q1b_ind_excluded_features, axis=1)
X_val = np.delete(X_val, q1b_ind_excluded_features, axis=1)
X_test = np.delete(X_test, q1b_ind_excluded_features, axis=1)
D = X_train.shape[1]


############################ Question 2 ##################################
def fit_linreg(X, yy, alpha):
    # data augmentation
    D = X.shape[1]
    N = X.shape[0]

    reg = np.sqrt(alpha) * np.eye(D, D)
    X1 = np.concatenate((X, np.ones((N, 1))), axis=1)
    reg1 = np.concatenate((reg, np.zeros((reg.shape[1], 1))), axis=1)
    X_aug = np.concatenate((X1, reg1), axis=0)
    y_aug = np.concatenate((yy, np.zeros((D, 1))), axis=0)

    # lstsq
    W, SSE, rank, singulars = np.linalg.lstsq(X_aug, y_aug, rcond=None)
    W_lstsq = W[:-1]
    b_lstsq = W[-1]
    return W_lstsq, b_lstsq

# least square method
alpha = 10
W_lstsq, b_lstsq = fit_linreg(X_train, y_train, alpha)

# gradient method
alpha = 10
W_grad, b_grad = fit_linreg_gradopt(X_train, np.squeeze(y_train), alpha)


# Errors
def compute_RMSE(X, y, w, b):
    # expand_dims to all single dimensional arrays
    if len(y.shape) == 1:
        y = np.expand_dims(y, -1)

    if len(w.shape) == 1:
        w = np.expand_dims(w, -1)

    # compute RMSE
    y_bar = np.dot(X, w) + b
    square_erros = np.square(y_bar - y)
    RMSE = np.sqrt(np.mean(square_erros))
    return RMSE


q2_RMSE_lstsq_tr = compute_RMSE(X_train, y_train, W_lstsq, b_lstsq)
q2_RMSE_lstsq_val = compute_RMSE(X_val, y_val, W_lstsq, b_lstsq)
q2_RMSE_lstsq_test = compute_RMSE(X_test, y_test, W_lstsq, b_lstsq)

q2_RMSE_grad_tr = compute_RMSE(X_train, y_train, W_grad, b_grad)
q2_RMSE_grad_val = compute_RMSE(X_val, y_val, W_grad, b_grad)
q2_RMSE_grad_test = compute_RMSE(X_test, y_test, W_grad, b_grad)


############################ Question 3 ##################################
# Question 3i
def fit_and_measure_on_projection(K):
    alpha = 10
    proj_mat = random_proj(D, K)

    # projected X
    X_train_proj = np.dot(X_train, proj_mat)
    X_val_proj = np.dot(X_val, proj_mat)
    X_test_proj = np.dot(X_test, proj_mat)

    results = {"K": K}

    # fitting
    W_lstsq_proj, b_lstsq_proj = fit_linreg(X_train_proj, y_train, alpha)
    W_grad_proj, b_grad_proj = fit_linreg_gradopt(X_train_proj, np.squeeze(y_train), alpha)

    # RMSE
    results['RMSE_lstsq_tr'] = compute_RMSE(X_train_proj, y_train, W_lstsq_proj, b_lstsq_proj)
    results['RMSE_lstsq_val'] = compute_RMSE(X_val_proj, y_val, W_lstsq_proj, b_lstsq_proj)
    results['RMSE_lstsq_test'] = compute_RMSE(X_test_proj, y_test, W_lstsq_proj, b_lstsq_proj)

    results['RMSE_grad_tr'] = compute_RMSE(X_train_proj, y_train, W_grad_proj, b_grad_proj)
    results['RMSE_grad_val'] = compute_RMSE(X_val_proj, y_val, W_grad_proj, b_grad_proj)
    results['RMSE_grad_test'] = compute_RMSE(X_test_proj, y_test, W_grad_proj, b_grad_proj)

    return results


K = 10
q3a_results_proj_10 = fit_and_measure_on_projection(K)

K = 100
q3a_results_proj_100 = fit_and_measure_on_projection(K)

K = 373 * 3
q3a_results_proj_373 = fit_and_measure_on_projection(K)

# Question 3ii
_save_filename_png = os.path.abspath("./presentation/presentation_figures/fig_01.pdf")
plt.figure()
plt.title("Histogram of feature 46")
plt.hist(X_train[45].ravel(), bins=30)
plt.xlabel('value')
plt.ylabel('number of samples')
plt.savefig(_save_filename_png)
plt.show()

q3b_pcg = np.sum(np.logical_or(X_train == 0, X_train < 0)) / X_train.size


def fit_and_measure_added_binaries():
    alpha = 10

    # projected X
    def aug_fn(X): return np.concatenate([X, X == 0, X < 0], axis=1)

    X_train_aug = aug_fn(X_train)
    X_val_aug = aug_fn(X_val)
    X_test_aug = aug_fn(X_test)

    # fitting
    W_lstsq, b_lstsq = fit_linreg(X_train_aug, y_train, alpha)
    # W_grad_proj, b_grad_proj = fit_linreg_gradopt(X_train_proj, np.squeeze(y_train), alpha)

    # RMSE
    results = {}
    results['RMSE_lstsq_tr'] = compute_RMSE(X_train_aug, y_train, W_lstsq, b_lstsq)
    results['RMSE_lstsq_val'] = compute_RMSE(X_val_aug, y_val, W_lstsq, b_lstsq)
    results['RMSE_lstsq_test'] = compute_RMSE(X_test_aug, y_test, W_lstsq, b_lstsq)

    # results['RMSE_grad_tr'] = compute_RMSE(X_train_proj, y_train, W_grad_proj, b_grad_proj)
    # results['RMSE_grad_val'] = compute_RMSE(X_val_proj, y_val, W_grad_proj, b_grad_proj)
    # results['RMSE_grad_test'] = compute_RMSE(X_test_proj, y_test, W_grad_proj, b_grad_proj)

    return results


q3b_results_added_binaries = fit_and_measure_added_binaries()

############################ Question 4 ##################################
# fit each class
K = 10  # number of thresholded classification problems to fit
mx = np.max(y_train)
mn = np.min(y_train)
hh = (mx - mn) / (K + 1)
thresholds = np.linspace(mn + hh, mx - hh, num=K, endpoint=True)

alpha = 10
weight_dict = {}
for kk in range(K):
    labels = y_train > thresholds[kk]

    # fit logistic regression
    ww, bb = fit_logreg_gradopt(X_train, np.squeeze(labels), alpha)
    weight_dict[kk] = {}
    weight_dict[kk]['w'] = ww
    weight_dict[kk]['b'] = bb

# create X_smart_proj
weights = []
bias = []
for key, value in weight_dict.items():
    weights.append(value["w"])
    bias.append(value["b"])
ww = np.stack(weights, axis=1)
bb = np.expand_dims(np.stack(bias), 0)


def sigmoid(x): return 1 / (1 + np.exp(-x))


X_train_smart = sigmoid(np.dot(X_train, ww) + bb)
X_val_smart = sigmoid(np.dot(X_val, ww) + bb)
X_test_smart = sigmoid(np.dot(X_test, ww) + bb)

# X_train_smart_1 = np.dot(X_train, ww) + bb
# X_val_smart_1 = np.dot(X_val, ww) + bb

alpha = 10
W_smart, b_smart = fit_linreg(X_train_smart, y_train, alpha)

q4_RMSE_smart_tr = compute_RMSE(X_train_smart, y_train, W_smart, b_smart)
q4_RMSE_smart_val = compute_RMSE(X_val_smart, y_val, W_smart, b_smart)
q4_RMSE_smart_test = compute_RMSE(X_test_smart, y_test, W_smart, b_smart)

############################ Question 5 ##################################
# random init
np.random.seed(1)
init_params = (np.random.randn(10), np.array(0), np.random.randn(10, D), np.zeros(10))
ww1, bb1, V1, bk1 = fit_cnn_gradopt(X_train, np.squeeze(y_train), 10, init_params)
params1 = (ww1, bb1, V1, bk1)

# sophisticated init
init_params = (np.squeeze(W_smart), np.squeeze(b_smart), ww.T, np.squeeze(bb))
ww2, bb2, V2, bk2 = fit_cnn_gradopt(X_train, np.squeeze(y_train), 10, init_params)
params2 = (ww2, bb2, V2, bk2)


def compute_RMSE_cnn(X, y, params):
    y_bar = np.expand_dims(nn_cost(params, X), -1)
    square_error = np.square(y_bar - y)
    RMSE = np.sqrt(np.mean(square_error))
    return RMSE


q5_RMSE_rand_tr = compute_RMSE_cnn(X_train, y_train, params1)
q5_RMSE_rand_val = compute_RMSE_cnn(X_val, y_val, params1)
q5_RMSE_rand_test = compute_RMSE_cnn(X_test, y_test, params1)

q5_RMSE_soph_tr = compute_RMSE_cnn(X_train, y_train, params2)
q5_RMSE_soph_val = compute_RMSE_cnn(X_val, y_val, params2)
q5_RMSE_soph_test = compute_RMSE_cnn(X_test, y_test, params2)

# Question 6
def fit_and_RMSE_cnn(s, k, add_binaries, inputs):
    def aug_fn(X): return np.concatenate([X, X == 0, X < 0], axis=1)

    X_train = inputs['X_train']
    X_val = inputs['X_val']
    X_test = inputs['X_test']
    y_train = inputs['y_train']
    y_val = inputs['y_val']
    y_test = inputs['y_test']
    D = X_train.shape[1]

    # random init
    alpha = 10
    if add_binaries:
        X_train = aug_fn(X_train)
        X_val = aug_fn(X_val)
        X_test = aug_fn(X_test)
        D = 3*D

    init_params = (np.random.randn(k), np.array(0), np.random.randn(k, D), np.zeros(k))
    ww1, bb1, V1, bk1 = fit_cnn_gradopt(X_train + np.random.standard_normal(X_train.shape) * s,
                                        np.squeeze(y_train), alpha, init_params)

    params1 = (ww1, bb1, V1, bk1)

    results = {}
    results['RMSE_tr'] = compute_RMSE_cnn(X_train, y_train, params1)
    results['RMSE_val'] = compute_RMSE_cnn(X_val, y_val, params1)
    results['RMSE_test'] = compute_RMSE_cnn(X_test, y_test, params1)
    return results, params1

sig = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6]
K = [10, 20, 40, 60]
add_binaries = [True, False]

inputs = {'X_train': X_train,
          'X_val': X_val,
          'X_test': X_test,
          'y_train': y_train,
          'y_val': y_val,
          'y_test': y_test}

# results = []
# params = []
# hyperparams = []
# for ii, s in enumerate(sig):
#     results.append([])
#     params.append([])
#     hyperparams.append([])
#     for jj, k in enumerate(K):
#         results[ii].append([])
#         params[ii].append([])
#         hyperparams[ii].append([])
#         for kk, bin in enumerate(add_binaries):
#             results[ii][jj].append([])
#             params[ii][jj].append([])
#             hyperparams[ii][jj].append([])
#             for n in range(5):
#                 tmp = fit_and_RMSE_cnn(s,k,bin,inputs)
#                 results[ii][jj][kk].append(tmp[0])
#                 params[ii][jj][kk].append(tmp[1])
#                 hyperparams[ii][jj][kk].append({'s':s, 'k':k, 'add_binaries':bin, 'n':n})

fm = open('./results.p', 'rb')
q6_results = pickle.load(fm)

RMSE_errors = np.array(q6_results['results'])
def choose_error(x): return x['RMSE_tr']
RMSE_tr = np.vectorize(choose_error)(RMSE_errors)
RMSE_tr_mean = RMSE_tr.mean(-1)
RMSE_tr_sterror = RMSE_tr.std(-1, ddof=1)/np.sqrt(5)
def choose_error(x): return x['RMSE_val']
RMSE_val = np.vectorize(choose_error)(RMSE_errors)
RMSE_val_mean = RMSE_val.mean(-1)
RMSE_val_sterror = RMSE_val.std(-1, ddof=1)/np.sqrt(5)
def choose_error(x): return x['RMSE_test']
RMSE_test = np.vectorize(choose_error)(RMSE_errors)
RMSE_test_mean = RMSE_test.mean(-1)
RMSE_test_sterror = RMSE_test.std(-1, ddof=1)/np.sqrt(5)

q6_argmin = np.unravel_index(np.argmin(RMSE_val_mean, axis=None), RMSE_val_mean.shape)

plt.figure()
for i, k in enumerate([10, 20, 40, 60]):
    plt.errorbar(sig, RMSE_val_mean[:, i, 1], RMSE_val_sterror[:, i, 1], label='K='+str(k), marker='o', linestyle='None')
    # plt.plot(sig, RMSE_val_mean[:, i, 1], 'o-', label = str(k))
plt.axhline(y=0.25213, xmin=0.05, xmax=0.95, linewidth=1, color = 'k')
plt.legend()
plt.xlabel('added noise')
plt.ylabel('RMSE')
plt.title('Validation Error')
# plt.yticks([RMSE_val_mean[argmin]])
plt.savefig('./presentation/presentation_figures/val_error.pdf')
plt.show()

