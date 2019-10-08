from scipy import io
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.io.wavfile import write

# set path
filepath = '/home/givaisile/OR_DS_data/MLPR/audio/amp_data.mat'

# configuration
D = 21

file = io.loadmat(filepath)
amp_data = file['amp_data']

plt.figure(); plt.plot(amp_data[:, 0], 'r'); plt.show(block=False);
# plt.hist(amp_data, bins=200); plt.show(block=False);


remaining_samples = amp_data.shape[0] % D
X = amp_data[:-remaining_samples] if remaining_samples > 0 else amp_data
X = X.reshape((int(X.shape[0]/D), D))
X_initial = copy.deepcopy(X)

# prepare dataset
np.random.shuffle(X)
index_tr = np.int(np.floor(X.shape[0] * 0.7))
index_val = np.int(np.floor(X.shape[0] * 0.15))
index_test = np.int(np.floor(X.shape[0] * 0.15))

train_start = 0
train_stop = index_tr + 1
val_start = index_tr + 1
val_stop = index_tr+1+index_val
test_start = index_tr+1+index_val
test_stop = index_tr+1+2*index_val


def split_x(X, start, stop):
    return X[start:stop, :20]


def split_y(X, start, stop):
    return X[start:stop, 20]


X_shuf_train = split_x(X, train_start, train_stop)
y_shuf_train = split_y(X, train_start, train_stop)

X_shuf_val = split_x(X, val_start, val_stop)
y_shuf_val = split_y(X, val_start, val_stop)

X_shuf_test = split_x(X, test_start, test_stop)
y_shuf_test = split_y(X, test_start, test_stop)


# question 2 - plot in example
i = 1000

plt.plot(np.linspace(0, 19/20, 20), X_shuf_train[i], 'r-o');
plt.plot(1, y_shuf_train[i], 'g-o');
plt.show(block=False);

time = np.linspace(0, 19 / 20, 20)
time = np.expand_dims(time, -1)
time_expanded = np.concatenate((time, np.ones_like(time)), axis=1)
y_tmp = X_shuf_train[i]


w, residuals, rank, sv = np.linalg.lstsq(time_expanded, y_tmp)

# subquestion a
t = 1
y_pred = w[0]*t + w[1]

plt.figure()
# real points
plt.plot(np.linspace(0, 19/20, 20), X_shuf_train[i], 'b-o', label = 'real points')
# real label
plt.plot(1, y_shuf_train[i], 'r-o', label = 'real label')
# fit line
t_gen = np.linspace(-.1, 1.1, 200)
plt.plot(t_gen, w[0]*t_gen + w[1], 'g-', label = 'fit line')
# prediction on 20/20
plt.plot(1, w[0]*1 + w[1], 'y-o', label = 'prediction')
plt.xlabel("time")
plt.ylabel("amplitudes")
plt.legend();
plt.show(block=False)


x_tmp = np.concatenate((np.ones_like(time), time, time**2, time**3, time**4), axis=1)
w, residuals, rank, sv = np.linalg.lstsq(x_tmp, y_tmp)

t = 1
y_pred = w[0]*t + w[1]

plt.figure()
# real points
plt.plot(np.linspace(0, 19/20, 20), X_shuf_train[i], 'b-o', label = 'real points')
# real label
plt.plot(1, y_shuf_train[i], 'r-o', label = 'real label')
# fit line
t_gen = np.expand_dims(np.linspace(-.1, 1.1, 200), -1)
t_gen = np.concatenate((np.ones_like(t_gen), t_gen, t_gen**2, t_gen**3, t_gen**4), axis=1)
plt.plot(t_gen, np.matmul(t_gen, w), 'g-', label = 'fit polynomial')
# prediction on 20/20
plt.plot(1, w[0]*1 + w[1], 'y-o', label = 'prediction')
plt.xlabel("time")
plt.ylabel("amplitudes")
plt.legend();
plt.show(block=False)

