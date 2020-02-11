from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import umap
from pyro.nn import PyroSample, PyroModule
import pyro.distributions as dist
from torch import nn


iris = load_iris()
print(iris.DESCR)

reducer = umap.UMAP()
reducer.fit(iris.data)
embedding = reducer.transform(iris.data)


class BayesianRegression(PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))

    def forward(self, x, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        mean = self.linear(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean

plt.figure()
plt.scatter(embedding[:, 0], embedding[:, 1], c=iris.target)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar()
plt.title('UMAP projection of the Digits dataset', fontsize=12)
plt.show(block=False)
