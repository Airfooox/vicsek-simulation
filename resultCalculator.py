import multiprocessing as mp
import json
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

    absoluteVelocity = 0
    for i in subSimulationRange:
        with open(dirOfData + '_' + str(i) + '/absoluteVelocity.txt') as resultFile:
            absoluteVelocity += json.load(resultFile)

    absoluteVelocity = absoluteVelocity / len(subSimulationRange)

    dict[num] = {'va': absoluteVelocity, 'constants': constants}

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

    for entry in sortedDict:
        index = entry[0]
        result = entry[1]
        # print(result['constants']['numSwimmers'], result['va'])
        # x.append(result['constants']['numSwimmers']/(result['constants']['environmentSideLength'])**2)
        x.append(result['constants']['randomAngleAmplitude'])
        # x.append(result['constants']['oscillationAmplitude'])
        y.append(result['va'])

    obj = {
        'x': x,
        'y': y
    }

    with open(dir + '/plotResult.txt', 'w') as plotFile:
        json.dump(obj, plotFile)