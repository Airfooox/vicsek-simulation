import multiprocessing as mp
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def calculateVelocity(dir):
    with open(dir + '/constants.txt') as constantsFile:
        constants = json.load(constantsFile)
    timeSteps = constants["time"] * constants["fps"]
    numSwimmers = constants["numSwimmers"]

    stateData = np.load(dir + '/statesData.npy')
    absoluteVelocity = 0
    for t in range(timeSteps):
        timeData = stateData[t]
        va = np.array([0, 0], float)
        for i in range(numSwimmers):
            swimmerData = timeData[i]
            va += np.array([np.cos(swimmerData[2]), np.sin(swimmerData[2])])
        absoluteVelocity += np.linalg.norm(va) / numSwimmers

    return absoluteVelocity / timeSteps

def calculateResult(calculationData):
    dir = calculationData[0]
    num = calculationData[1]
    dict = calculationData[2]

    dirOfData = dir + '/' + str(num)
    subSimulationRange = range(50)

    with open(dirOfData + '_1' + '/constants.txt') as constantsFile:
        constants = json.load(constantsFile)

    absoluteVelocity = 0
    for i in subSimulationRange:
        absoluteVelocity += calculateVelocity(dirOfData + '_' + str(i))
    absoluteVelocity = absoluteVelocity / len(subSimulationRange)

    dict[num] = {'va': absoluteVelocity, 'constants': constants}

if __name__ == "__main__":
    manager = mp.Manager()
    sameEtaDirectory = manager.dict()
    sameEtaRange = range(150)

    sameEtaPool = []
    for i in sameEtaRange:
        sameEtaPool.append(['E:/simulationdata/sameEtaGroup', i, sameEtaDirectory])

    pool = mp.Pool(processes=(mp.cpu_count() - 2))
    for _ in tqdm(pool.imap(func=calculateResult, iterable=sameEtaPool),
                  total=len(sameEtaPool)):
        pass
    pool.close()
    pool.join()

    def takeKeyToSort(element):
        return element[0]

    sortedDict = sorted(sameEtaDirectory.items(), key = lambda x : x[0])
    x, y = [], []

    for entry in sortedDict:
        index = entry[0]
        result = entry[1]
        # print(result['constants']['numSwimmers'], result['va'])
        x.append(result['constants']['numSwimmers']/(result['constants']['environmentSideLength'])**2)
        # x.append(result['constants']['randomAngleAmplitude'])
        y.append(result['va'])

    plt.plot(x, y, 'ro')
    # plt.xlabel(r'$\rho \longrightarrow$')
    plt.xlabel(r'$\eta \longrightarrow$')
    plt.ylabel(r'$v_a \longrightarrow$')
    plt.grid('both')
    plt.show()