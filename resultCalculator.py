import multiprocessing as mp
import json
import numpy as np
import matplotlib.pyplot as plt

def calculateVelocity(dir):
    constants = {}
    with open(dir + '/constants.txt') as constantsFile:
        constants = json.load(constantsFile)
    timeSteps = constants["time"] * constants["fps"]
    numSwimmers = constants["numSwimmers"]

    stateData = np.load(dir + '/statesData.npy')
    absoluteVelocity = np.array([0, 0], float)
    for t in range(timeSteps):
        timeData = stateData[t]
        for i in range(numSwimmers):
            swimmerData = timeData[i]
            absoluteVelocity += np.array([np.cos(swimmerData[2]), np.sin(swimmerData[2])])

    return np.linalg.norm(absoluteVelocity) / (timeSteps * numSwimmers)

def calculateResult(calculationData):
    dir = calculationData[0]
    num = calculationData[1]
    dict = calculationData[2]

    # print(calculationData)

    dirOfData = dir + '/' + str(num)
    subSimulationRange = range(1, 4)

    constants = {}
    with open(dirOfData + '_1' + '/constants.txt') as constantsFile:
        constants = json.load(constantsFile)

    absoluteVelocity = 0
    for i in subSimulationRange:
        absoluteVelocity += calculateVelocity(dirOfData + '_' + str(i))
    absoluteVelocity = absoluteVelocity / len(subSimulationRange)

    dict[num] = {'va': absoluteVelocity, 'constants': constants}

if __name__ == "__main__":
    manager = mp.Manager()
    sameRhoDirectory = manager.dict()
    # sameEtaRange = range(1, 501)
    sameRhoRange = range(0, 50)

    # calculateResult(['E:/simulationdata/sameRho', str(1), sameRhoDirectory])

    sameRhoPool = []
    for i in sameRhoRange:
        sameRhoPool.append(['D:/simulationdata/sameRho', i, sameRhoDirectory])

    pool = mp.Pool(processes=(mp.cpu_count() - 2))
    pool.map(calculateResult, sameRhoPool)
    pool.close()
    pool.join()

    def takeKeyToSort(element):
        return element[0]

    sortedDict = sorted(sameRhoDirectory.items(), key = lambda x : x[0])
    x, y = [], []

    for entry in sortedDict:
        index = entry[0]
        result = entry[1]
        print(index)
        x.append(result['constants']['randomAngleAmplitude'])
        y.append(result['va'])

    plt.plot(x, y, 'ro')
    plt.show()