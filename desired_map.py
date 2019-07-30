import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = fig.gca(projection='3d')

cRoordinates = [(0,0), (np.cos(np.pi/2), np.sin(np.pi/2)), (np.cos(-np.pi/6), np.sin(-np.pi/6)), (np.cos(7*np.pi/6), np.sin(7*np.pi/6))]

X = np.arange(-2, 2, 0.05)
Y = np.arange(-2, 2, 0.05)

X, Y = np.meshgrid(X, Y)
Z = X*0
Z_des = X*0

print(coordinates)
for x,y in coordinates:
    Z += np.exp(-((X-x)**2 / 0.1 + (Y-y)**2 / 0.1))
    x += 0.1
    Z_des += np.exp(-((X-x)**2 / 0.1 + (Y-y)**2 / 0.1))

# print(np.linalg.norm(Z - Z_des))

# surf = ax.plot_surface(X, Y, Z_des, cmap=cm.coolwarm)
plt.contourf(X, Y, Z, 30)
plt.savefig('fig.png', dpi=300)
