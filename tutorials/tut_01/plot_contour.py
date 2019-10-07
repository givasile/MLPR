import numpy as np
import matplotlib.pyplot as plt

v = np.expand_dims(np.array([1, 1]), -1)
b = 5

x = np.linspace(-10, 10, 1000)
y = np.linspace(-10, 10, 1000)
xx, yy = np.meshgrid(x, y)

z = 1 / (1 + np.exp(-(v[0]*xx + v[1]*yy + b)))


plt.contourf(xx, yy, z, 100, cmap='RdGy')
plt.colorbar()
plt.show(block=False)