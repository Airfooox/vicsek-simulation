# setup
from classes import Simulation
from util import printProgressBar
import numpy as np
import time

from simulationManager import SimulationManager, SimulationGroup

if __name__ == "__main__":
    starttime = time.perf_counter()

    simulationDir = '/local/kzisiadis/vicsek-simulation'
    # simulationDir = 'D:/simulationdata'
    simulationManager = SimulationManager(simulationDir)


    def sameEtaConstants(i, numSimulation, defaultTimeSteps):
        return {
            "timeSteps": 7200,
            "timePercentageUsedForMean": 25,

            "environmentSideLength": 7,
            "numSwimmers": i + 1,
            "interactionRadius": 1,
            "randomAngleAmplitude": 2,
            "oscillationAmplitude": np.pi / 40,
            "oscillationPeriod": 60,  # how many timesteps for one full oscillation

            "initialVelocity": 0.0025,
            "swimmerSize": 0.04,

            "saveVideo": False,
        }

    sameEtaGroup = SimulationGroup(simulationDataDir=simulationDir + '/sameEtaGroup', constantsFunc=sameEtaConstants, numSimulation=500, repeatNum=100, saveTrajectoryData=False)
    simulationManager.appendGroup(sameEtaGroup)

    def sameRhoConstants400(i, numSimulation, defaultTimeSteps):
        return {
            "timeSteps": 7200,
            "timePercentageUsedForMean": 25,

            "environmentSideLength": 10,
            "numSwimmers": 400,
            "interactionRadius": 1,
            "randomAngleAmplitude": 15 * (i / numSimulation),
            "oscillationAmplitude": np.pi / 40,
            # "oscillationAmplitude": 0,
            "oscillationPeriod": 60,  # how many timesteps for one full oscillation

            "initialVelocity": 0.0025,
            "swimmerSize": 0.04,

            "saveVideo": False,
        }


    sameRhoGroup400 = SimulationGroup(simulationDataDir=simulationDir + '/sameRhoGroup400',
                                      constantsFunc=sameRhoConstants400, numSimulation=100, repeatNum=50,
                                      saveTrajectoryData=False)
    simulationManager.appendGroup(sameRhoGroup400)

    def sameRhoConstants1000(i, numSimulation, defaultTimeSteps):
        return {
            "timeSteps": 7200,
            "timePercentageUsedForMean": 25,

            "environmentSideLength": 15.8113,
            "numSwimmers": 1000,
            "interactionRadius": 1,
            "randomAngleAmplitude": 15 * (i / numSimulation),
            # "oscillationAmplitude": np.pi / 40,
            "oscillationAmplitude": 0,
            "oscillationPeriod": 60,  # how many timesteps for one full oscillation

            "initialVelocity": 0.0025,
            "swimmerSize": 0.04,

            "saveVideo": False,
        }


    # sameRhoGroup1000 = SimulationGroup(simulationDataDir=simulationDir + '/sameRhoGroup1000WithoutOcsi',
    #                                   constantsFunc=sameRhoConstants1000, numSimulation=100, repeatNum=50,
    #                                   saveTrajectoryData=False)
    # simulationManager.appendGroup(sameRhoGroup1000)


    def sameRhoConstants4000(i, numSimulation, defaultTimeSteps):
        return {
            "timeSteps": 7200,
            "timePercentageUsedForMean": 25,

            "environmentSideLength": 31.6,
            "numSwimmers": 4000,
            "interactionRadius": 1,
            "randomAngleAmplitude": 15 * (i / numSimulation),
            # "oscillationAmplitude": np.pi / 40,
            "oscillationAmplitude": 0,
            "oscillationPeriod": 60, # how many timesteps for one full oscillation

            "initialVelocity": 0.0025,
            "swimmerSize": 0.04,

            "saveVideo": False,
        }


    # sameRhoGroup4000 = SimulationGroup(simulationDataDir=simulationDir + '/sameRhoGroup4000',
    #                                    constantsFunc=sameRhoConstants4000, numSimulation=100, repeatNum=50,
    #                                    saveTrajectoryData=False)
    # simulationManager.appendGroup(sameRhoGroup4000)

    simulationManager.simulate()
    print('That took {} seconds'.format(time.perf_counter() - starttime))
