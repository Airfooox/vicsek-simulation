import multiprocessing as mp
import json

import numpy
import numpy as np
from tqdm import tqdm

def calculateResult(calculationData):
    dir = calculationData[0]
    num = calculationData[1]
    dict = calculationData[2]

    dirOfData = dir + '/' + str(num)
    subSimulationRange = range(50)

    with open(dirOfData + '_0' + '/constants.txt') as constantsFile:
        constants = json.load(constantsFile)

    totalTimeSteps = constants['fps'] * constants['time']
    timeEvolution = np.zeros((totalTimeSteps), dtype=numpy.float64)

    absoluteVelocity = 0
    for i in subSimulationRange:
        with open(dirOfData + '_' + str(i) + '/absoluteVelocity.txt') as resultFile:
            absoluteVelocity += json.load(resultFile)

        timeEvolutionData = np.load(dirOfData + '_' + str(i) + '/absoluteVelocities.npy')
        timeEvolution = np.add(timeEvolution, timeEvolutionData)


        # with open(dirOfData + '_' + str(i) + '/absoluteVelocities.npy') as timeEvolutionFile:
        #     timeEvolutionData = np.load(timeEvolutionFile)
        #     np.add(timeEvolution, timeEvolutionData)

    absoluteVelocity = absoluteVelocity / len(subSimulationRange)
    timeEvolution = timeEvolution / len(subSimulationRange)

    dict[num] = {'va': absoluteVelocity, 'constants': constants, 'timeEvolution': timeEvolution.tolist()}

if __name__ == "__main__":
dir = '/local/kzisiadis/vicsek-simulation/sameRhoGroup1000'

    manager = mp.Manager()
    simulationGroupDirectory = manager.dict()
    simulationGroupRange = range(50)

    simulationGroupPool = []
    for i in simulationGroupRange:
        simulationGroupPool.append([dir, i, simulationGroupDirectory])

    pool = mp.Pool(processes=(mp.cpu_count() - 2))
    for _ in tqdm(pool.imap(func=calculateResult, iterable=simulationGroupPool),
                  total=len(simulationGroupPool)):
        pass
    pool.close()
    pool.join()

    def takeKeyToSort(element):
        return element[0]
    sortedDict = sorted(simulationGroupDirectory.items(), key = lambda x : x[0])
    x, y = [], []
    timeEvolution = []

    for entry in sortedDict:
        index = entry[0]
        result = entry[1]
        # print(result['constants']['numSwimmers'], result['va'])
        # x.append(result['constants']['numSwimmers']/(result['constants']['environmentSideLength'])**2)
        x.append(result['constants']['randomAngleAmplitude'])
        # x.append(result['constants']['oscillationAmplitude'])
        y.append(result['va'])
        timeEvolution.append(result['timeEvolution'])

    obj = {
        'x': x,
        'y': y
    }

    with open(dir + '/plotResult.txt', 'w') as plotFile:
        json.dump(obj, plotFile)

    with open(dir + '/timeEvolutionResult.txt', 'w') as timeEvolutionFile:
        json.dump(timeEvolution, timeEvolutionFile)