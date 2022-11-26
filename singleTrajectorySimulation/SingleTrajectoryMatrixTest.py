# append parent directory for importing
# https://codeolives.com/2020/01/10/python-reference-module-in-parent-directory/
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from Simulation import Simulation
from SimulationManager import SimulationManager
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # simulationDir = '/local/kzisiadis/vicsek-simulation/singleTrajectoryMatrixTest'
    simulationDir = 'D:/simulationdata/singleTrajectoryMatrixTest'
    simulationManager = SimulationManager(simulationDir)


    def singleSwimmerConstantsInitialParameter(simulationIndex, totalNumberOfSimulations, swimmerIndex):
        return {
            "oscillationAmplitude": np.pi / 64,
            "oscillationPeriod": 140,  # how many timesteps for one full oscillation
            "oscillationPhaseshift": 0
        }


    CONSTANTS = {
        "timeSteps": 500,
        "timePercentageUsedForMean": 25,

        "environmentSideLength": 2,
        "numSwimmers": 1,
        "interactionRadius": 1,
        "randomAngleAmplitude": 0,

        "velocity": 0.0025,
        "swimmerSize": 0.04,

        "saveVideo": False,
    }
    def singleSwimmerConstants(simulationIndex, totalNumberOfSimulations, defaultTimeSteps):
        return CONSTANTS



    simulation = Simulation(1, 1, CONSTANTS, singleSwimmerConstantsInitialParameter)
    simulationData = simulation.simulate()
    simulation.animate()

    x, y, phi = simulationData[:, 0, 0], simulationData[:, 0, 1], (180/np.pi) * simulationData[:, 0, 2]
    DeltaR = list(map(lambda xy: (xy[0] - x[0])**2 + (xy[1] - y[0])**2, zip(x, y)))
    # print(DeltaR)
    t = range(len(x))
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(t, DeltaR)
    ax1.set_title("$(\Delta R)^2$")
    ax1.grid("both")
    ax2.plot(t, phi, "go-", ms=2.5)
    ax2.set_title("$\phi$")
    ax2.set_yticks(np.arange(-270, 270, step=45))
    ax2.grid("both")

    plt.show()

    # singleSwimmerSimulation = SimulationGroup(simulationDataDir=simulationDir + '/singleSwimmer',
    #                                   constantsFunc=singleSwimmerConstants, initialParameterFunc = singleSwimmerConstantsInitialParameter,
    #                                   numSimulation=100, repeatNum=100,
    #                                   saveTrajectoryData=False)
    # simulationManager.appendGroup(singleSwimmerSimulation)

