# Siehe Seite 4:
# Erkundung der Interaktionen von Teilchen mit unterschiedlichen Phasen

# append parent directory for importing
# https://codeolives.com/2020/01/10/python-reference-module-in-parent-directory/
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import itertools
from Simulation import Simulation
import numpy as np

if __name__ == "__main__":
    # trajectoriesPictureDir = r'E:\simulationdata\singleTrajectories\eta=0'
    #
    # if not (os.path.exists(trajectoriesPictureDir) and os.path.isdir(trajectoriesPictureDir)):
    #     os.mkdir(trajectoriesPictureDir)

    simulationConfig = {
        "timeSteps": 350,

        "environmentSideLength": 1,
        "groups": {
            "1": {
                "numSwimmers": 1,
                "oscillationAmplitude": np.pi / 32,
                "oscillationPeriod": 100,  # how many timesteps for one full oscillation
                "oscillationPhaseShift": 0
            },
            "2": {
                "numSwimmers": 1,
                "oscillationAmplitude": np.pi / 32,
                "oscillationPeriod": 100,  # how many timesteps for one full oscillation
                "oscillationPhaseShift": np.pi / 2
            }
        },
        "interactionRadius": 0.2,
        "randomAngleAmplitude": 0,

        "velocity": 0.0025,
        "swimmerSize": 0.04,

        "saveVideo": False,
    }

    simulation = Simulation(simulationIndex=1, numSimulation=1, simulationConfig=simulationConfig,
                            timePercentageUsedForMean=25)
    # override start position and orientation
    print(simulation.states[0, 0, :])
    simulation.states[0, 0, :3] = np.array(
        [0.05 * simulationConfig['environmentSideLength'], 0.3 * simulationConfig['environmentSideLength'], 0],
        dtype=np.float64)
    print(simulation.states[0, 0, :])
    simulation.states[0, 1, :3] = np.array(
        [0.45 * simulationConfig['environmentSideLength'], 0.7 * simulationConfig['environmentSideLength'], 0],
        dtype=np.float64)
    simulation.initializeGrid()
    simulation.simulate()
    statesData = simulation.states

    videoPath = r'C:\Users\konst\OneDrive\Uni\Anstellung\Prof. Menzel (2020-22)\vicsek\simulation\videos'
    simulation.animate(showGroup=True, saveVideo=True, videoPath=os.path.join(videoPath, 'PhaseShiftInteractionTest.mp4'))

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