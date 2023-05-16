# append parent directory for importing
# https://codeolives.com/2020/01/10/python-reference-module-in-parent-directory/
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import itertools
from Simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
from util import multipleFormatter

if __name__ == "__main__":
    trajectoriesPictureDir = r'D:\simulationdata\singleTrajectories_eta=0'

    if not (os.path.exists(trajectoriesPictureDir) and os.path.isdir(trajectoriesPictureDir)):
        os.mkdir(trajectoriesPictureDir)

    simulationConfigs = [
        # {'eta': [0], 'amplitude': np.array([1/2, 1/4]) * np.pi, 'period': [30, 50]},
        # {'eta': [0], 'amplitude': np.array([1/8, 1/16, 1/32, 1/64]) * np.pi, 'period': np.arange(10, 70+1, 20)},
        {'eta': [0], 'amplitude': [np.pi / 4], 'period': [50]}
    ]

    for configEntry in simulationConfigs:
        for config in itertools.product(configEntry['eta'], configEntry['amplitude'], configEntry['period']):
            eta = config[0]
            amplitude = config[1]
            period = config[2]

            simulationConfig = {
                "timeSteps": 500,

                "environmentSideLength": 2,
                "groups": {
                    "1": {
                        "numSwimmers": 1,
                        "snakingAmplitude": amplitude,
                        "snakingPeriod": period,  # how many timesteps for one full oscillation
                        "snakingPhaseshift": 0
                    }
                },
                "interactionRadius": 1,
                "randomAngleAmplitude": eta,
                'interactionStrengthFactor': 0,

                "velocity": 0.03,
                "swimmerSize": 0.04,

                "saveVideo": False,
            }

            simulation = Simulation(simulationIndex=1, numSimulation=1, simulationConfig=simulationConfig,
                                    timePercentageUsedForMean=25)
            # override start position and orientation
            simulation.states[0, 0, :3] = np.array([0.05 * simulationConfig['environmentSideLength'], 0.5 * simulationConfig['environmentSideLength'], 0], dtype=np.float64)
            simulation.initializeGrid()
            simulation.simulate()
            statesData = simulation.states

            # saveFixedTimeSetPictureDir = os.path.join(trajectoriesPictureDir,
            #              f'singleSwimmerTrajectory_timeSteps={simulationConfig["timeSteps"]}_eta={eta}_amplitude={np.round(amplitude / np.pi, 3)}pi_period={period}.png')
            # simulation.animate(fixedTimeStep=simulationConfig['timeSteps'] - 1, saveFixedTimeSetPictureDir=saveFixedTimeSetPictureDir)

            # simulation.animate(fixedTimeStep=simulationConfig['timeSteps'] - 1, saveFixedTimeSetPictureDir=None)
            simulation.animate(fixedTimeStep=None)

            # x, y, phi = statesData[:, 0, 0], statesData[:, 0, 1], (180/np.pi) * statesData[:, 0, 2]
            # DeltaR = list(map(lambda xy: (xy[0] - x[0])**2 + (xy[1] - y[0])**2, zip(x, y)))

            # plt.figure()
            # trajectoryLine, = plt.plot(statesData[:, 0, 0], statesData[:, 0, 1], "go", ms=0.75)
            # t = np.arange(len(x))
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # ax1.plot(t, DeltaR)
            # ax1.set_title("$(\Delta R)^2$")
            # ax1.grid("both")
            # ax2.plot(t, phi, "go-", ms=2.5)
            # ax2.set_title("$\phi$")
            # ax2.set_yticks(np.arange(-270, 270, step=45))
            # ax2.grid("both")

            # plt.show()

