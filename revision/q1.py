import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# generate points
N = 100
sigma = 1
X1 = stats.multivariate_normal(mean=[-5, -5], cov=np.eye(2)).rvs(N)
X2 = stats.multivariate_normal(mean=[5, 5], cov=np.eye(2)).rvs(N)
Y1 = np.zeros(N)
Y2 = np.zeros(N)

X3 = stats.multivariate_normal(mean=[-5, 5], cov=np.eye(2)).rvs(N)
X4 = stats.multivariate_normal(mean=[5, -5], cov=np.eye(2)).rvs(N)
Y3 = np.ones(N)
Y4 = np.ones(N)

# plot
plt.figure()
plt.scatter(X1[:,0], X1[:,1], c='r')
plt.scatter(X2[:,0], X2[:,1], c='r')
plt.scatter(X3[:,0], X3[:,1], c='b')
plt.scatter(X4[:,0], X4[:,1], c='b')
plt.show(block=False)


X = np.concatenate([X1, X2, X3, X4], axis=0)
tmp2 = np.stack([np.zeros(2*N), np.ones(2*N)], axis = -1)
tmp1 =  np.stack([np.ones(2*N), np.zeros(2*N)], axis = -1)
Y = np.concatenate([tmp1,tmp2], axis=0)

results = np.linalg.lstsq(np.concatenate([X, np.ones([X.shape[0],1])], axis=1), Y)



# hand derived features
tmp1 = stats.multivariate_normal(mean=[-5, -5], cov=np.eye(2)*10).pdf(X)
tmp2 = stats.multivariate_normal(mean=[5, 5], cov=np.eye(2)*10).pdf(X)
tmp3 = stats.multivariate_normal(mean=[-5, 5], cov=np.eye(2)*10).pdf(X)
tmp4 = stats.multivariate_normal(mean=[5, -5], cov=np.eye(2)*10).pdf(X)

phi_X = np.stack([tmp1 + tmp2, tmp3+tmp4], axis=1)

plt.figure()
plt.scatter(phi_X[:2*N,0], phi_X[:2*N,1], c='r')
plt.scatter(phi_X[2*N:,0], phi_X[2*N:,1], c='b')
plt.show(block=False)

results1 = np.linalg.lstsq(np.concatenate([phi_X, np.ones([phi_X.shape[0],1])], axis=1), Y)
