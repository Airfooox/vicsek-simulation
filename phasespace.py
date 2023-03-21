import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("TkAgg")

import numpy as np
from scipy.integrate import odeint
from scipy.misc import derivative

v0 = 0.03
eta = 0.4
A = np.pi/8
T = 50
phi = 0

t = 0

def f(vec, t):
    x, y = vec
    return [
        v0 * (x * np.cos(A * np.cos((2*np.pi/T) * t + phi))- y * np.sin(A * np.cos((2*np.pi/T) * t + phi))) - x,
        v0 * (y * np.cos(A * np.cos((2 * np.pi / T) * t + phi)) + x * np.sin(A * np.cos((2 * np.pi / T) * t + phi))) - y
    ]

xLinSpace = np.linspace(-5, 5, 20)
yLinSpace = np.linspace(-5, 8, 30)
xSpace, ySpace = np.meshgrid(xLinSpace, yLinSpace)

u, v = np.zeros(xSpace.shape), np.zeros(ySpace.shape)

NI, NJ = xSpace.shape
for i in range(NI):
    for j in range(NJ):
        x = xSpace[i, j]
        y = ySpace[i, j]
        df = f([x, y], t)
        u[i, j] = df[0]
        v[i, j] = df[1]

trajectoryTimeLinSpace = np.linspace(0, 1500, 1500)
vec0 = [1, 1]
sol = odeint(f, vec0, trajectoryTimeLinSpace)

plt.quiver(xSpace, ySpace, u, v, color='r')
plt.quiver(sol[:-1, 0], sol[:-1, 1], sol[1:, 0]-sol[:-1, 0], sol[1:, 1]-sol[:-1, 1], scale_units='xy', angles='xy', scale=1, color='green')
plt.xlabel(r'$x \longrightarrow$')
plt.ylabel(r'$y \longrightarrow$')
plt.show()