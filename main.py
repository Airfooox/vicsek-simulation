# setup
from classes import Simulation
from util import printProgressBar
import numpy as np
import time

from simulationManager import SimulationManager, SimulationGroup


def constantFunc(environmentSideLength=7, numSwimmers=1, randomAngleAmplitude=2, oscillationAmplitude=np.pi / 40):
    return {
        "fps": 30,
        "time": 60,
        "environmentSideLength": environmentSideLength,
        "numSwimmers": numSwimmers,
        "initialVelocity": 0.1,
        "interactionRadius": 0.1,
        "randomAngleAmplitude": randomAngleAmplitude,  # eta
        "oscillationAmplitude": oscillationAmplitude,

        "swimmerSize": 0.04,
        "saveVideo": False
    }


if __name__ == "__main__":
    starttime = time.time()

    simulationManager = SimulationManager('E:/simulationdata')


    def sameEtaConstants(i, numSimulation, time, fps):
        return {
            "time": time,
            "fps": fps,

            "environmentSideLength": 7,
            "numSwimmers": i + 1,
            "interactionRadius": 1,
            "randomAngleAmplitude": 2,
            "oscillationAmplitude": np.pi / 40,

            "initialVelocity": 0.075,
            "swimmerSize": 0.04,
            "saveVideo": False
        }


    sameEtaGroup = SimulationGroup('E:/simulationdata/sameEtaGroup', sameEtaConstants, 150, 50)

    simulationManager.appendGroup(sameEtaGroup)
    simulationManager.simulate()

    # constantsArraySameEtaConfigs = []
    # for i in range(1, 101):
    #     constantsConfigWithoutOscillation = constantFunc(4, i, 2, 0)
    #     constantsConfigWithOscillation = constantFunc(4, i, 2)
    #
    #     constantsArraySameEtaConfigs.append(['E:/simulationdata/r = 0,1/sameEtaWOOscillation', str(i) + '_1', constantsConfigWithoutOscillation])
    #     constantsArraySameEtaConfigs.append(['E:/simulationdata/r = 0,1/sameEtaWOOscillation', str(i) + '_2', constantsConfigWithoutOscillation])
    #     constantsArraySameEtaConfigs.append(['E:/simulationdata/r = 0,1/sameEtaWOOscillation', str(i) + '_3', constantsConfigWithoutOscillation])
    #     constantsArraySameEtaConfigs.append(['E:/simulationdata/r = 0,1/sameEtaWOOscillation', str(i) + '_4', constantsConfigWithoutOscillation])
    #     constantsArraySameEtaConfigs.append(['E:/simulationdata/r = 0,1/sameEtaWOOscillation', str(i) + '_5', constantsConfigWithoutOscillation])
    #
    #     constantsArraySameEtaConfigs.append(['E:/simulationdata/r = 0,1/sameEtaWOscillation', str(i) + '_1', constantsConfigWithOscillation])
    #     constantsArraySameEtaConfigs.append(['E:/simulationdata/r = 0,1/sameEtaWOscillation', str(i) + '_2', constantsConfigWithOscillation])
    #     constantsArraySameEtaConfigs.append(['E:/simulationdata/r = 0,1/sameEtaWOscillation', str(i) + '_3', constantsConfigWithOscillation])
    #     constantsArraySameEtaConfigs.append(['E:/simulationdata/r = 0,1/sameEtaWOscillation', str(i) + '_4', constantsConfigWithOscillation])
    #     constantsArraySameEtaConfigs.append(['E:/simulationdata/r = 0,1/sameEtaWOscillation', str(i) + '_5', constantsConfigWithOscillation])
    #
    # constantsArraySameRhoProcesses = []
    # for i in range(0, 101):
    #     constantsConfigWithoutOscillation = constantFunc(5, 100, 5 * (i / 100), 0)
    #     constantsConfigWithOscillation = constantFunc(5, 100, 5 * (i / 100))
    #
    #     constantsArraySameRhoProcesses.append(['E:/simulationdata/r = 0,1/sameRhoWOOscillation', str(i) + '_1', constantsConfigWithoutOscillation])
    #     constantsArraySameRhoProcesses.append(['E:/simulationdata/r = 0,1/sameRhoWOOscillation', str(i) + '_2', constantsConfigWithoutOscillation])
    #     constantsArraySameRhoProcesses.append(['E:/simulationdata/r = 0,1/sameRhoWOOscillation', str(i) + '_3', constantsConfigWithoutOscillation])
    #     constantsArraySameRhoProcesses.append(['E:/simulationdata/r = 0,1/sameRhoWOOscillation', str(i) + '_4', constantsConfigWithoutOscillation])
    #     constantsArraySameRhoProcesses.append(['E:/simulationdata/r = 0,1/sameRhoWOOscillation', str(i) + '_5', constantsConfigWithoutOscillation])
    #
    #     constantsArraySameRhoProcesses.append(['E:/simulationdata/r = 0,1/sameRhoWOscillation', str(i) + '_1', constantsConfigWithOscillation])
    #     constantsArraySameRhoProcesses.append(['E:/simulationdata/r = 0,1/sameRhoWOscillation', str(i) + '_2', constantsConfigWithOscillation])
    #     constantsArraySameRhoProcesses.append(['E:/simulationdata/r = 0,1/sameRhoWOscillation', str(i) + '_3', constantsConfigWithOscillation])
    #     constantsArraySameRhoProcesses.append(['E:/simulationdata/r = 0,1/sameRhoWOscillation', str(i) + '_4', constantsConfigWithOscillation])
    #     constantsArraySameRhoProcesses.append(['E:/simulationdata/r = 0,1/sameRhoWOscillation', str(i) + '_5', constantsConfigWithOscillation])
    #
    # poolConfigMerge = constantsArraySameEtaConfigs + constantsArraySameRhoProcesses
    # pool = mp.Pool(processes=(mp.cpu_count() - 2))
    # pool.map(runSimulation, poolConfigMerge)
    # pool.close()
    #
    # print('That took {} seconds'.format(time.time() - starttime))

    # runSimulation('./results/sameRho', 1, CONSTANTS )

    # constants = constantFunc(31.5, 4000, 0.5)
    # simulation = Simulation(constants)
    # printProgressBar(0, simulation.numFrames, prefix='Simulation Progress:', suffix='Simulation Complete', length=50)
    # simulation.simulate()
    # states = simulation.states
    print('That took {} seconds'.format(time.time() - starttime))
    # simulation.animate()
