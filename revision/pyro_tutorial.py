import pyro.distributions as dist
import pandas as pd
from pyro import sample
from pyro import condition
from pyro.poutine import trace
import pyro
import matplotlib.pyplot as plt

pyro.clear_param_store()


def sleep_model():
    # Very likely to feel lazy
    lazy = sample("lazy", dist.Bernoulli(0.9))
    if lazy:
        # Only going to (possibly) ignore my alarm if I'm feeling lazy
        alarm = sample("alarm", dist.Bernoulli(0.8))
        # Will sleep more if I ignore my alarm
        sleep = sample("sleep", dist.Normal(8 + 2 * alarm, 1))
    else:
        sleep = sample("sleep", dist.Normal(6, 1))
    return sleep


tr = trace(sleep_model).get_trace()

cond_model = condition(sleep_model, {"lazy": 1., "alarm": 0., "sleep": 10.})


# trace(cond_model).get_trace().log_prob_sum().exp()

# traces = []
# for _ in range(1000):
#     tr = trace(sleep_model).get_trace()
#     values = {
#         name: props['value'].item()
#         for (name, props) in tr.nodes.items()
#         if props['type'] == 'sample'
#     }
#     traces.append(values)


# pd.DataFrame(traces).hist()
# plt.show()
