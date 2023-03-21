import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("TkAgg")

import numpy as np

N = 2
eta = np.float64(0.45 * 0)
A = np.array([np.pi/16, np.pi/16], dtype=np.float64) * 0
T = np.array([50, 50], dtype=np.float64)
phi = np.array([0, np.pi/2], dtype=np.float64)

interactionStrengthFactor = np.float64(1)

tMax = 750
x, y, theta = np.zeros((tMax, N), dtype=np.float64), np.zeros((tMax, N), dtype=np.float64), np.zeros((tMax, N), dtype=np.float64)
# theta[0, :] = (theta[0, :]  + 1) * A * np.cos(2*np.pi / T * 0 + phi)

x[0, 0], y[0, 0], theta[0, 0] = 0, 1, -np.pi/4
x[0, 1], y[0, 1], theta[0, 1] = 0, 0, np.pi/4

xInt1, xInt2, yInt1, yInt2 = None, None, None, None
for t in range(1, tMax):
    for i in range(N):
        # theta[t] = theta[t-1] + A * np.cos((2*np.pi / T) * t + phi[i])

        sinSum, cosSum = np.sin(theta[t-1, i]), np.cos(theta[t-1, i])
        for j in range(N):
            if i == j:
                continue
            r = np.sqrt((x[t-1, i] - x[t-1, j])**2 + (y[t-1, i] - y[t-1, j])**2)
            if r > 0.2:
                continue
            if xInt1 == None:
                xInt1 = x[t-1, i]
                xInt2 = x[t-1, j]
                yInt1 = y[t-1, i]
                yInt2 = y[t-1, j]
            sinSum += np.sin(theta[t-1, j])
            cosSum += np.cos(theta[t-1, j])
        averageAngle = np.arctan2(sinSum, cosSum)
        randomAngle = ((np.random.rand() - 0.5) * eta)
        cosinesOscillation = A[i] * np.cos((2 * np.pi / T[i]) * t + phi[i])
        theta[t, i] = averageAngle + randomAngle + cosinesOscillation

        # factor = interactionStrengthFactor / np.pi
        # sum = 0
        # for j in range(N):
        #     if i == j:
        #         continue
        #     r = np.sqrt((x[t-1, i] - x[t-1, j])**2 + (y[t-1, i] - y[t-1, j])**2)
        #     if r > 0.2:
        #         continue
        #     if xInt1 == None:
        #         xInt1 = x[t-1, i]
        #         xInt2 = x[t-1, j]
        #         yInt1 = y[t-1, i]
        #         yInt2 = y[t-1, j]
        #     sum += -np.sin(theta[t-1, i] - theta[t-1, j])
        # randomAngle = ((np.random.rand() - 0.5) * eta)
        # cosinesOscillation = A[i] * np.cos((2 * np.pi / T[i]) * t + phi[i])
        # theta[t, i] = theta[t-1, i] + factor * sum + randomAngle + cosinesOscillation


        xVelCos = np.cos(theta[t, i])
        yVelSin = np.sin(theta[t, i])
        x[t, i] = x[t-1, i] + 0.003 * xVelCos
        y[t, i] = y[t-1, i] + 0.003 * yVelSin

# fig1 = plt.figure(1)
# plt.plot(np.arange(tMax), theta)
# plt.xlabel(r'$t \longrightarrow$')
# plt.ylabel(r'$\theta \longrightarrow$')

# fig2 = plt.figure(2)
for i in range(N):
    plt.plot(x[:, i], y[:, i])
plt.xlabel(r'$x \longrightarrow$')
plt.ylabel(r'$y \longrightarrow$')
plt.xlim((0.1, 0.9))
plt.ylim((0.1, 0.9))
plt.vlines(xInt1, 0, 1, colors=['g'], linestyles='dashed', linewidth=1)
# plt.vlines(xInt2, 0, 1, colors=['orange'], linestyles='dashed', linewidth=1)
plt.hlines(yInt1, 0, 1,  colors=['g'], linestyles='dashed', linewidth=1)
plt.hlines(yInt2, 0, 1,  colors=['g'], linestyles='dashed', linewidth=1)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
ax.xaxis.get_label().set_fontsize(14)
ax.yaxis.get_label().set_fontsize(14)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

# inset axes....
axins = ax.inset_axes([0.7, 0.7, 0.27, 0.27])
axins.plot(x[:, 0], y[:, 0])
axins.vlines(xInt1, 0, 1, colors=['g'], linestyles='dashed', linewidth=0.25)
axins.hlines([yInt1, yInt2], 0, 1,  colors=['g'], linestyles='dashed', linewidth=0.25)
# subregion of the original image
x1, x2, y1, y2 = 0.385, 0.415, 0.585, 0.615
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels([])
axins.set_yticklabels([])
ax.indicate_inset_zoom(axins, edgecolor="black")

plt.show()