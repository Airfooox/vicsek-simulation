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


    # def sameEtaConstants(i, numSimulation, time, fps):
    #     return {
    #         "time": time,
    #         "fps": fps,
    #
    #         "environmentSideLength": 7,
    #         "numSwimmers": i + 1,
    #         "interactionRadius": 1,
    #         "randomAngleAmplitude": 2,
    #         "oscillationAmplitude": np.pi / 40,
    #
    #         "initialVelocity": 0.075,
    #         "swimmerSize": 0.04,
    #
    #         "saveVideo": False,
    #     }
    #
    #
    # sameEtaGroup = SimulationGroup(simulationDataDir=simulationDir + '/sameEtaGroup', constantsFunc=sameEtaConstants, numSimulation=500, repeatNum=50, saveTrajectoryData=False)
    # simulationManager.appendGroup(sameEtaGroup)

    def sameRhoConstants400(i, numSimulation, time, fps):
        return {
            "time": time,
            "fps": fps,

            "environmentSideLength": 10,
            "numSwimmers": 400,
            "interactionRadius": 1,
            "randomAngleAmplitude": 5 * (i / numSimulation),
            # "oscillationAmplitude": np.pi / 40,
            "oscillationAmplitude": 0,

            "initialVelocity": 0.075,
            "swimmerSize": 0.04,

            "saveVideo": False,
        }


    sameRhoGroup400 = SimulationGroup(simulationDataDir=simulationDir + '/sameRhoGroup400',
                                      constantsFunc=sameRhoConstants400, numSimulation=50, repeatNum=50,
                                      saveTrajectoryData=False)
    # simulationManager.appendGroup(sameRhoGroup400)

    def sameRhoConstants1000(i, numSimulation, time, fps):
        return {
            "time": 7 * time,
            "fps": fps,

            "environmentSideLength": 15.8113,
            "numSwimmers": 1000,
            "interactionRadius": 1,
            "randomAngleAmplitude": 5 * (i / numSimulation),
            # "oscillationAmplitude": np.pi / 40,
            "oscillationAmplitude": 0,

            "initialVelocity": 0.075,
            "swimmerSize": 0.04,

            "saveVideo": False,
        }


    sameRhoGroup1000 = SimulationGroup(simulationDataDir=simulationDir + '/sameRhoGroup1000',
                                      constantsFunc=sameRhoConstants1000, numSimulation=50, repeatNum=50,
                                      saveTrajectoryData=False)
    simulationManager.appendGroup(sameRhoGroup1000)


    def sameRhoConstants4000(i, numSimulation, time, fps):
        return {
            "time": 10 * time,
            "fps": fps,

            "environmentSideLength": 31.6,
            "numSwimmers": 4000,
            "interactionRadius": 1,
            "randomAngleAmplitude": 5 * (i / numSimulation),
            # "oscillationAmplitude": np.pi / 40,
            "oscillationAmplitude": 0,

            "initialVelocity": 0.075,
            "swimmerSize": 0.04,

            "saveVideo": False,
        }


    sameRhoGroup4000 = SimulationGroup(simulationDataDir=simulationDir + '/sameRhoGroup4000',
                                       constantsFunc=sameRhoConstants4000, numSimulation=50, repeatNum=50,
                                       saveTrajectoryData=False)
    # simulationManager.appendGroup(sameRhoGroup4000)

    simulationManager.simulate()

    # constants = sameEtaConstants(1, 1, 15, 1)
    # simulation = Simulation(constants)
    # # printProgressBar(0, simulation.numFrames, prefix='Simulation Progress:', suffix='Simulation Complete', length=50)
    # simulation.simulate()
    # simulation.getAbsoluteVelocityTotal()
    # states = simulation.states
    print('That took {} seconds'.format(time.perf_counter() - starttime))
    # simulation.animate()
