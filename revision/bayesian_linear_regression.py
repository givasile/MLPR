import numpy as np
import torch
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt
from torch.distributions import constraints
pyro.clear_param_store()

guess = 15.
measurement = 1.5
s_weight = 7
s_measurement = .75


# P(m,w;g)
def scale(guess):
    weight = pyro.sample("weight", pyro.distributions.Normal(guess, s_weight))
    return pyro.sample("measurement", pyro.distributions.Normal(weight, s_measurement))


def scale_parametrized_guide(guess):
    a = pyro.param("a", torch.tensor(guess))
    b = pyro.param("b", torch.tensor(1.), constraint=constraints.positive)
    return pyro.sample("weight", dist.Normal(a, torch.abs(b)))


def perfect_guide(guess):
    loc = (s_measurement**2 * guess + measurement) / (s_weight + s_measurement**2)
    scale = np.sqrt(s_measurement**2/(s_weight + s_measurement**2))  # 0.6
    return pyro.sample("weight", dist.Normal(loc, scale)), loc, scale


def plot_distribution(dist, guess):
    x = []
    for i in range(1000):
        x.append(dist(guess).tolist())
    plt.figure()
    plt.hist(x, bins=50, range=(0, 15), density=True)
    plt.show(block=False)


# P(w|m;g)
conditioned_scale = pyro.condition(scale, data={"measurement": measurement})


svi = pyro.infer.SVI(model=conditioned_scale,
                     guide=scale_parametrized_guide,
                     optim=pyro.optim.SGD({"lr": 0.001, "momentum":0.1}),
                     loss=pyro.infer.Trace_ELBO())


# plt.figure()
# plt.plot(losses)
# plt.title("ELBO")
# plt.xlabel("step")
# plt.ylabel("loss")
# plt.show(block=False)

# _, a_correct, b_correct = perfect_guide(guess)
# print('a_correct = ', a_correct)
# print('a = ', pyro.param("a").item())
# print('b_correct = ', b_correct)
# print('b = ', pyro.param("b").item())

# plt.figure()
# plt.subplot(1, 2, 1)
# plt.plot([0, num_steps], [a_correct, a_correct], 'k:')
# plt.plot(a)
# plt.ylabel('a')

# plt.subplot(1, 2, 2)
# plt.ylabel('b')
# plt.plot([0, num_steps], [b_correct, b_correct], 'k:')
# plt.plot(b)
# plt.tight_layout()
# plt.show(block=False)
