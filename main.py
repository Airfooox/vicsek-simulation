# setup
from classes import Simulation
from util import printProgressBar
import multiprocessing as mp
import os
import json
import numpy as np

def constantFunc(environmentSideLength=7, numSwimmers=1, randomAngleAmplitude=2):
    return {
        "fps": 60,
        "time": 60,
        "environmentSideLength": environmentSideLength,
        "numSwimmers": numSwimmers,
        "initialVelocity": 0.25,
        "interactionRadius": 0.3,
        "randomAngleAmplitude": randomAngleAmplitude,  # eta

        "swimmerSize": 0.04,
        "saveVideo": False
    }

def runSimulation(simulationData):
    dir = simulationData[0]
    num = simulationData[1]
    constants = simulationData[2]

    simulation = Simulation(constants)
    simulation.simulate()
    statesData = simulation.states

    dirOfData = dir + '/' + str(num)
    os.mkdir(dirOfData)
    with open(dirOfData + '/constants.txt', 'w') as constantsFile:
        json.dump(constants, constantsFile)
    np.save(dirOfData + '/statesData', statesData)

if __name__ == "__main__":


    # constant rho
    constantsArraySameEtaConfigs = []
    for i in range(500):
        constants = constantFunc(7, i, 2)

        constantsArraySameEtaConfigs.append(['E:/simulationdata/sameEta', str(i) + '_1', constants])
        constantsArraySameEtaConfigs.append(['E:/simulationdata/sameEta', str(i) + '_2', constants])
        constantsArraySameEtaConfigs.append(['E:/simulationdata/sameEta', str(i) + '_3', constants])
        constantsArraySameEtaConfigs.append(['E:/simulationdata/sameEta', str(i) + '_4', constants])
        constantsArraySameEtaConfigs.append(['E:/simulationdata/sameEta', str(i) + '_5', constants])

    constantsArraySameRhoProcesses = []
    for i in range(500):
        constants = constantFunc(6, 50, 5 * (i / 500))

        constantsArraySameEtaConfigs.append(['E:/simulationdata/sameRho', str(i) + '_1', constants])
        constantsArraySameEtaConfigs.append(['E:/simulationdata/sameRho', str(i) + '_2', constants])
        constantsArraySameEtaConfigs.append(['E:/simulationdata/sameRho', str(i) + '_3', constants])
        constantsArraySameEtaConfigs.append(['E:/simulationdata/sameRho', str(i) + '_4', constants])
        constantsArraySameEtaConfigs.append(['E:/simulationdata/sameRho', str(i) + '_5', constants])

    poolConfigMerge = constantsArraySameEtaConfigs + constantsArraySameRhoProcesses
    pool = mp.Pool(processes=16)
    pool.map(runSimulation, constantsArraySameEtaConfigs)
    pool.close()

    # runSimulation('./results/sameRho', 1, CONSTANTS )
    # simulation = Simulation(CONSTANTS)
    # printProgressBar(0, simulation.numFrames, prefix='Simulation Progress:', suffix='Simulation Complete', length=50)
    # simulation.simulate()
    # states = simulation.states
    # simulation.animate()
