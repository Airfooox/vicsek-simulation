import os
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("TkAgg")

import numpy as np

import itertools
from Simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
from util import multipleFormatter

if __name__ == "__main__":
    trajectoriesPictureDir = r'C:\Users\konst\OneDrive\Uni\Lehre\7. Semester\Bachelorarbeit\tex\img\singleTrajectories_eta=0'

    if not (os.path.exists(trajectoriesPictureDir) and os.path.isdir(trajectoriesPictureDir)):
        os.mkdir(trajectoriesPictureDir)

    simulationConfigs = [
        {'eta': [0], 'amplitude': np.array([1/2]) * np.pi, 'period': [30]},
        # {'eta': [0], 'amplitude': np.array([1/2, 1/4]) * np.pi, 'period': [30, 50]},
        # {'eta': [0], 'amplitude': np.array([1/8]) * np.pi, 'period': np.arange(10, 70+1, 20)},
        # {'eta': [0], 'amplitude': np.array([1/16]) * np.pi, 'period': np.arange(10, 110+1, 20)},
        # {'eta': [0], 'amplitude': np.array([1/32]) * np.pi, 'period': np.arange(10, 170+1, 20)},
        # {'eta': [0], 'amplitude': np.array([1/64]) * np.pi, 'period': np.arange(10, 250+1, 20)},
        # {'eta': [0], 'amplitude': [np.pi / 32], 'period': [130]}
    ]

    for configEntry in simulationConfigs:
        for config in itertools.product(configEntry['eta'], configEntry['amplitude'], configEntry['period']):
            eta = config[0]
            amplitude = config[1]
            period = config[2]

            phi = 0

            tMax = 150
            theta = np.zeros(tMax)
            x, y = np.zeros(tMax), np.zeros(tMax)

            theta[0] = amplitude * np.cos(phi)

            # k1 = amplitude * np.cos(phi)
            # k2 = amplitude * np.cos((2*np.pi / T) * 1/2 + phi)
            # k3 = amplitude * np.cos((2*np.pi / T) * 1/2 + phi)
            # k4 = amplitude * np.cos((2*np.pi / T) * 1 + phi)
            # theta[0] = (k1 + 2*k2 + 2*k3 + k4) / 6

            x[0], y[0] = 0, 0
            for t in range(1, tMax):
                # theta[t] = theta[t-1] + amplitude * np.cos((2*np.pi / period) * t + phi)
                averageAngle = np.arctan2(np.sin(theta[t - 1]), np.cos(theta[t - 1]))
                # averageAngle = theta[t-1]
                randomAngle = ((np.random.rand() - 0.5) * eta)
                # cosinesOscillation = amplitude * np.cos((2 * np.pi / period) * t + phi)

                k1 = amplitude * np.cos((2*np.pi / period) * t + phi)
                k2 = amplitude * np.cos((2*np.pi / period) * (t+1/2) + phi)
                k3 = amplitude * np.cos((2*np.pi / period) * (t+1/2) + phi)
                k4 = amplitude * np.cos((2*np.pi / period) * (t+1) + phi)
                cosinesOscillation = (k1 + 2*k2 + 2*k3 + k4) / 6

                theta[t] = averageAngle + randomAngle + cosinesOscillation
                xVelCos = np.cos(theta[t])
                yVelSin = np.sin(theta[t])
                x[t] = x[t - 1] + 0.03 * xVelCos
                y[t] = y[t - 1] + 0.03 * yVelSin

            # print(theta[0], theta[1])
            # fig1 = plt.figure(1)
            # plt.plot(np.arange(tMax), theta)
            # plt.xlabel(r'$t \longrightarrow$')
            # plt.ylabel(r'$\theta \longrightarrow$')

            fig2, axis2 = plt.subplots()
            plt.plot(x, y)
            # plt.xlim((-0.1, 7.2))
            # plt.ylim((-2.1, 2.8))
            plt.xlabel(r'$x \longrightarrow$')
            plt.ylabel(r'$y \longrightarrow$')
            axis2.set(adjustable='datalim', aspect='equal')

            xStart, xEnd = axis2.get_xlim()
            yStart, yEnd = axis2.get_ylim()
            # axis2.xaxis.set_ticks(np.round(np.arange(xStart, xEnd, (xEnd - xStart + 1) / 5), 2))
            plt.xticks(fontsize=11 * 1.5)
            plt.yticks(fontsize=11 * 1.5)
            axis2.xaxis.get_label().set_fontsize(14 * 1.5)
            axis2.yaxis.get_label().set_fontsize(14 * 1.5)


            plt.show()

            # saveFixedTimeSetPictureDir = os.path.join(trajectoriesPictureDir,
            #                                           f'singleSwimmerTrajectory_timeSteps={tMax}_eta={eta}_amplitude={np.round(amplitude / np.pi, 4)}pi_period={period}.png')
            # plt.savefig(saveFixedTimeSetPictureDir, bbox_inches='tight')
            # plt.close()



