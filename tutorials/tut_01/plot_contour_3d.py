import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

v = np.expand_dims(np.array([1, 1, 1]), -1)
b = 0
start = -1
stop = 1
points = 20

x = np.linspace(start, stop, points)
y = np.linspace(start, stop, points)
z = np.linspace(start, stop, points)
xx, yy, zz = np.meshgrid(x, y, z)

h = 1 / (1 + np.exp(-(v[0]*xx + v[1]*yy + v[2]*zz + b)))

fig = plt.figure()
ax = plt.axes(projection = "3d")
img = ax.scatter(xx.flat, yy.flat, zz.flat, c=h.flat, cmap='RdGy')
fig.colorbar(img)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show(block=False)