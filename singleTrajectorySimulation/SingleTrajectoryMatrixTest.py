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

if __name__ == "__main__":
    trajectoriesPictureDir = r'E:\simulationdata\singleTrajectories\eta=0'

    if not (os.path.exists(trajectoriesPictureDir) and os.path.isdir(trajectoriesPictureDir)):
        os.mkdir(trajectoriesPictureDir)

    simulationConfigs = [
        {'eta': [0], 'amplitude': [np.pi / 16 * 0], 'period': [40]}
    ]

    for configEntry in simulationConfigs:
        for config in itertools.product(configEntry['eta'], configEntry['amplitude'], configEntry['period']):
            eta = config[0]
            amplitude = config[1]
            period = config[2]

            simulationConfig = {
                "timeSteps": 250,

                "environmentSideLength": 1,
                "groups": {
                    "1": {
                        "numSwimmers": 1,
                        "oscillationAmplitude": amplitude,
                        "oscillationPeriod": period,  # how many timesteps for one full oscillation
                        "oscillationPhaseShift": 0
                    },
                    # "2": {
                    #     "numSwimmers": 200,
                    #     "oscillationAmplitude":  8 * np.pi / 16,
                    #     "oscillationPeriod": 40,  # how many timesteps for one full oscillation
                    #     "oscillationPhaseShift": np.pi / 2
                    # }
                },
                "interactionRadius": 1,
                "randomAngleAmplitude": eta,

                "velocity": 0.0025,
                "swimmerSize": 0.04,

                "saveVideo": False,
            }

            simulation = Simulation(simulationIndex=1, numSimulation=1, simulationConfig=simulationConfig,
                                    timePercentageUsedForMean=25)
            # override start position and orientation
            simulation.states[0, 0, :3] = np.array([0.1, 0.5, 0], dtype=np.float64)
            simulation.simulate()
            statesData = simulation.states
            simulation.animate(fixedTimeStep=simulationConfig['timeSteps'] - 1)
            x, y, phi = statesData[:, 0, 0], statesData[:, 0, 1], (180/np.pi) * statesData[:, 0, 2]
            DeltaR = list(map(lambda xy: (xy[0] - x[0])**2 + (xy[1] - y[0])**2, zip(x, y)))

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

