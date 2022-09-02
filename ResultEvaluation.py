import multiprocessing as mp
import json
import os

import numpy as np
from numba import njit
from scipy.optimize import curve_fit
from tqdm import tqdm

@njit
def getMeanAbsolutVelocity(absoluteVelocities, startMean):
    arr = absoluteVelocities[startMean:]
    mean = 0
    for absoluteVelocity in arr:
        mean += absoluteVelocity
    return mean / len(arr)

def nonLinearFitting(x, y, func, initialParameters):
    # curve fit the test data
    fittedParameters, pcov = curve_fit(func, x, y, initialParameters, maxfev=10**4)


    # modelPredictions = func(x, *fittedParameters)

    # absError = modelPredictions - y

    # SE = np.square(absError)  # squared errors
    # MSE = np.mean(SE)  # mean squared errors
    # RMSE = np.sqrt(MSE)  # Root Mean Squared Error, RMSE
    # Rsquared = 1.0 - (np.var(absError) / np.var(y))
    # return {'parameters': fittedParameters, 'RMSE': RMSE, 'Rsquared': Rsquared}
    return {'parameters': fittedParameters}

def calculateAbsoluteVelocityTotal(timeSteps, framesUsedForMean, absoluteVelocities):
    absoluteVelocities = np.array(absoluteVelocities)
    absoluteVelocities[np.abs(absoluteVelocities) < np.finfo(float).eps] = 0
    # defining exponential function for saturation
    def func(t, a, b):
        return -a * (np.exp(-b * t) - 1)

    # find parameters for saturation function
    t = np.array(range(timeSteps))
    initialParameters = [0.5, 0.1]

    try:
        model = nonLinearFitting(t / len(absoluteVelocities), absoluteVelocities, func, initialParameters)
        a, b = model['parameters'][0], model['parameters'][1] / len(absoluteVelocities)

        # find the time when system is in saturation for getting the mean value of absolut velocities
        yprimelim = 10 ** (-5)
        saturationBorder = np.round(np.maximum(1/b * np.log(a * b / yprimelim), 0))
        if saturationBorder > framesUsedForMean:
            return -3
    except RuntimeError:
        return -1

    try:
        absolutVelocity = getMeanAbsolutVelocity(np.array(absoluteVelocities), framesUsedForMean)
    except ZeroDivisionError:
        absolutVelocity = -2
        print(a, b, framesUsedForMean, ' / ', timeSteps)
        # print(timeSteps, np.array(absoluteVelocities))
    return absolutVelocity

def calculateResult(calculationData):
    dir = calculationData[0]
    num = calculationData[1]
    dict = calculationData[2]
    timePercentageUsedForMean = calculationData[3]
    reevaluateAbsoluteVelocities = calculationData[4]

    dirOfData = dir + '/' + str(num)
    subSimulationRange = range(50)

    with open(dirOfData + '_0' + '/constants.txt') as constantsFile:
        constants = json.load(constantsFile)

    timeSteps = constants['timeSteps']
    framesUsedForMean = np.ceil(((1 - (constants["timePercentageUsedForMean"] / 100)) * constants["timeSteps"]))
    timeEvolution = np.zeros((timeSteps), dtype=np.float64)

    selectedSubVelocities = []
    for i in subSimulationRange:
        timeEvolutionData = np.load(dirOfData + '_' + str(i) + '/absoluteVelocities.npy')
        timeEvolution = np.add(timeEvolution, timeEvolutionData)

        if reevaluateAbsoluteVelocities:
            reevalutedAbsoluteVelocity = calculateAbsoluteVelocityTotal(timeSteps, framesUsedForMean, timeEvolutionData)
            with open(dirOfData + '_' + str(i) + '/absoluteVelocity.txt', 'w') as absoluteVelocityFile:
                json.dump(reevalutedAbsoluteVelocity, absoluteVelocityFile)

        with open(dirOfData + '_' + str(i) + '/absoluteVelocity.txt') as resultFile:
            absoluteVelocitySub = json.load(resultFile)

            if (absoluteVelocitySub > 0):
                selectedSubVelocities.append(absoluteVelocitySub)

            # try:
            #
            # except json.decoder.JSONDecodeError:
            #     print(dirOfData + '_' + str(i) + '/absoluteVelocity.txt')

        with open(dirOfData + '_' + str(i) + '/timeEvolution.txt', 'w') as timeEvolutionFile:
            json.dump(timeEvolutionData.tolist(), timeEvolutionFile)

    absoluteVelocity = np.mean(selectedSubVelocities)
    std = np.std(selectedSubVelocities)
    timeEvolution = timeEvolution / len(selectedSubVelocities)

    # absoluteVelocity = 0
    # ignoreCount = 0
    # usedVelocities = []
    # for i in subSimulationRange:
    #     timeEvolutionData = np.load(dirOfData + '_' + str(i) + '/absoluteVelocities.npy')
    #     timeEvolution = np.add(timeEvolution, timeEvolutionData)
    #
    #     if reevaluateAbsoluteVelocities:
    #         reevalutedAbsoluteVelocity = calculateAbsoluteVelocityTotal(timeSteps, timeEvolutionData)
    #         with open(dirOfData + '_' + str(i) + '/absoluteVelocity.txt', 'w') as absoluteVelocityFile:
    #             json.dump(reevalutedAbsoluteVelocity, absoluteVelocityFile)
    #
    #     with open(dirOfData + '_' + str(i) + '/absoluteVelocity.txt') as resultFile:
    #         absoluteVelocitySub = json.load(resultFile)
    #
    #         if (absoluteVelocitySub <= 0):
    #             print(absoluteVelocitySub, dirOfData + '_' + str(i) + '/absoluteVelocity.txt')
    #             ignoreCount += 1
    #         else:
    #             absoluteVelocity += absoluteVelocitySub
    #             usedVelocities.append(absoluteVelocitySub)
    #
    #         # try:
    #         #
    #         # except json.decoder.JSONDecodeError:
    #         #     print(dirOfData + '_' + str(i) + '/absoluteVelocity.txt')
    #
    #
    #
    #     with open(dirOfData + '_' + str(i) + '/timeEvolution.txt', 'w') as timeEvolutionFile:
    #         json.dump(timeEvolutionData.tolist(), timeEvolutionFile)

    # absoluteVelocity = absoluteVelocity / (len(subSimulationRange) - ignoreCount)
    # timeEvolution = timeEvolution / (len(subSimulationRange) - ignoreCount)
    # std = np.std(usedVelocities)

    dict[num] = {'va': absoluteVelocity, 'std': std, 'constants': constants, 'timeEvolution': timeEvolution.tolist()}

if __name__ == "__main__":
    dir = '/local/kzisiadis/vicsek-simulation/sameRhoGroup400'
    timePercentageUsedForMean = 25
    reevaluateAbsoluteVelocities = True

    manager = mp.Manager()
    simulationGroupDirectory = manager.dict()
    simulationGroupRange = range(100)

    simulationGroupPool = []
    for i in simulationGroupRange:
        simulationGroupPool.append([dir, i, simulationGroupDirectory, timePercentageUsedForMean, reevaluateAbsoluteVelocities])

    pool = mp.Pool(processes=(mp.cpu_count() - 2))
    for _ in tqdm(pool.imap(func=calculateResult, iterable=simulationGroupPool),
                  total=len(simulationGroupPool)):
        pass
    pool.close()
    pool.join()

    sortedDict = sorted(simulationGroupDirectory.items(), key = lambda x : x[0])
    x, y, std = [], [], []
    timeEvolution = []

    for entry in sortedDict:
        index = entry[0]
        result = entry[1]
        # print(result['constants']['numSwimmers']/((result['constants']['environmentSideLength'])**2), result['va'])
        # x.append(result['constants']['numSwimmers']/((result['constants']['environmentSideLength'])**2))
        x.append(result['constants']['randomAngleAmplitude'])
        # x.append(result['constants']['oscillationAmplitude'])
        y.append(result['va'])
        std.append(result['std'])
        timeEvolution.append(result['timeEvolution'])

    obj = {
        'x': x,
        'y': y,
        'std': std
    }

    with open(dir + '/plotResult.txt', 'w') as plotFile:
        json.dump(obj, plotFile)

    with open(dir + '/timeEvolutionResult.txt', 'w') as timeEvolutionFile:
        json.dump(timeEvolution, timeEvolutionFile)