import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool
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
    fittedParameters, pcov = curve_fit(func, x, y, initialParameters, maxfev=10 ** 4)

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
        saturationBorder = np.round(np.maximum(1 / b * np.log(a * b / yprimelim), 0))
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
    simulationGroupName = calculationData[0]
    simulationGroupPath = calculationData[1]
    currentSimulationNum = calculationData[2]
    simulationRepeatNum = calculationData[3]
    timePercentageUsedForMean = calculationData[4]
    simulationGroupDirectory = calculationData[5]
    reevaluateAbsoluteVelocities = calculationData[6]

    dirOfData = os.path.join(simulationGroupPath, str(currentSimulationNum))

    if not os.path.exists(simulationGroupPath):
        print(f'simulationGroup {simulationGroupPath} doesnt exist')
        return

    constants = None
    subSimulationPaths = []
    for i in range(simulationRepeatNum):
        simulationIdentifier = dirOfData + '_' + str(i)

        if (os.path.exists(simulationIdentifier) and
                os.path.exists(os.path.join(simulationIdentifier, 'totalVelocities.json')) and
                os.path.exists(os.path.join(simulationIdentifier, 'config.json'))):
            subSimulationPaths.append(simulationIdentifier)
            if constants is None:
                with open(os.path.join(simulationIdentifier, 'config.json')) as constantsFile:
                    constants = json.load(constantsFile)

    if constants is None or len(subSimulationPaths) == 0:
        print(f'simulation {dirOfData} has no data saved.')
        return

    timeSteps = constants['timeSteps']
    numberOfGroups = len(constants['groups'])
    framesUsedForMean = int(np.ceil(((1 - (timePercentageUsedForMean / 100)) * timeSteps)))
    timeEvolution = np.zeros((timeSteps), dtype=np.float64)
    timeEvolutionVectorialGroupVelocityDifferences = np.zeros((timeSteps, numberOfGroups-1), dtype=np.float64)
    vectorialGroupVelocityDifferences = np.zeros((len(subSimulationPaths), numberOfGroups-1), dtype=np.float64)

    selectedTotalAbsoluteVelocity = []
    selectedTotalAbsoluteGroupVelocities = []
    selectedTotalVectorialVelocity = []
    selectedTotalVectorialGroupVelocities = []
    selectedTotalNematicOrderParameter = []
    selectedTotalNematicOrderParameterGroups = []

    vectorialGroupDifferencesBuffer = np.zeros((len(subSimulationPaths), timeSteps, numberOfGroups - 1), dtype=np.float64)
    vectorialGroupDifferencesVariances = np.zeros((len(subSimulationPaths), numberOfGroups-1), dtype=np.float64)
    for subSimulationIndex, subSimulationPath in enumerate(subSimulationPaths):
        timeEvolutionData = np.load(os.path.join(subSimulationPath, 'absoluteVelocities.npy'))
        timeEvolution = np.add(timeEvolution, timeEvolutionData)

        vectorialGroupVelocities = np.load(os.path.join(subSimulationPath, 'vectorialGroupVelocities.npy'))

        for groupId in range(numberOfGroups - 1):
            for t in range(timeSteps):
                previousV = vectorialGroupVelocities[t, groupId]
                currentV = vectorialGroupVelocities[t, groupId + 1]

                dot = previousV[0] * currentV[0] + previousV[1] * currentV[1]
                det = previousV[0] * currentV[1] - previousV[1] * currentV[0]

                # calculate the angle between the inner vectors (so it only goes from 0 to 180°)
                innerAngle = np.abs(np.arctan2(det, dot))
                vectorialGroupDifferencesBuffer[subSimulationIndex, t, groupId] = innerAngle
            vectorialGroupVelocityDifferences[subSimulationIndex, groupId] = np.mean(vectorialGroupDifferencesBuffer[subSimulationIndex, framesUsedForMean:, :], axis=0)
            vectorialGroupDifferencesVariances[subSimulationIndex, groupId] = np.var(vectorialGroupDifferencesBuffer[subSimulationIndex, framesUsedForMean:, :], axis=0)

        timeEvolutionVectorialGroupVelocityDifferences = np.add(timeEvolutionVectorialGroupVelocityDifferences, vectorialGroupDifferencesBuffer)

        if reevaluateAbsoluteVelocities:
            reevalutedAbsoluteVelocity = calculateAbsoluteVelocityTotal(timeSteps, framesUsedForMean, timeEvolutionData)
            with open(os.path.join(subSimulationPath, 'totalVelocities.json'), 'rw') as totalVelocitiesFile:
                totalVelocities = json.load(totalVelocitiesFile)
                totalVelocities['totalAbsoluteVelocity'] = reevalutedAbsoluteVelocity
                json.dump(totalVelocities, totalVelocitiesFile)

        with open(os.path.join(subSimulationPath, 'totalVelocities.json')) as resultFile:
            totalVelocities = json.load(resultFile)

            if totalVelocities['totalAbsoluteVelocity'] > 0:
                selectedTotalAbsoluteVelocity.append(totalVelocities['totalAbsoluteVelocity'])
                selectedTotalAbsoluteGroupVelocities.append(totalVelocities['totalAbsoluteGroupVelocities'])
                selectedTotalVectorialVelocity.append(totalVelocities['totalVectorialVelocity'])
                selectedTotalVectorialGroupVelocities.append(totalVelocities['totalVectorialGroupVelocities'])
                selectedTotalNematicOrderParameter.append(totalVelocities['totalNematicOrderParameter'])
                selectedTotalNematicOrderParameterGroups.append(totalVelocities['totalNematicOrderParameterGroups'])

            # try:
            #
            # except json.decoder.JSONDecodeError:
            #     print(dirOfData + '_' + str(i) + '/totalVelocities.txt')

        with open(os.path.join(subSimulationPath, 'timeEvolution.json'), 'w') as timeEvolutionFile:
            json.dump(timeEvolutionData.tolist(), timeEvolutionFile)

    timeEvolution = timeEvolution / len(selectedTotalAbsoluteVelocity)
    timeEvolutionVectorialGroupVelocityDifferences = timeEvolutionVectorialGroupVelocityDifferences / len(subSimulationPaths)

    totalAbsoluteVelocity, std = np.mean(selectedTotalAbsoluteVelocity), np.std(selectedTotalAbsoluteVelocity)
    totalAbsoluteGroupVelocities, totalAbsoluteGroupVelocityStds = np.mean(selectedTotalAbsoluteGroupVelocities, axis=0), np.std(selectedTotalAbsoluteGroupVelocities, axis=0)

    totalNematicOrderParameter, totalNematicOrderParameterStd = np.mean(selectedTotalNematicOrderParameter), np.std(selectedTotalNematicOrderParameter)
    totalNematicOrderParameterGroups, totalNematicOrderParameterGroupsStd = np.mean(selectedTotalNematicOrderParameterGroups, axis=0), np.std(selectedTotalNematicOrderParameterGroups, axis=0)

    vectorialGroupVelocityDifferencesMean = np.mean(vectorialGroupVelocityDifferences, axis=1)[0]
    varianceSum = np.sum(vectorialGroupDifferencesVariances)
    covarianceSum = 0
    for k in range(len(subSimulationPaths)):
        for j in range(k):
            Xk = vectorialGroupDifferencesBuffer[k, framesUsedForMean:, 0]
            Xj = vectorialGroupDifferencesBuffer[j, framesUsedForMean:, 0]
            covarianceSum += np.cov(Xk, Xj)[0][1]
    vectorialGroupVelocityDifferencesStd = (varianceSum + 2*covarianceSum) / len(subSimulationPaths)**2

    simulationGroupDirectory[currentSimulationNum] = {
        'absoluteVelocities': selectedTotalAbsoluteVelocity,
        'totalAbsoluteVelocity': totalAbsoluteVelocity,
        'totalAbsoluteVelocityStd': std,
        'subAbsoluteGroupVelocities': selectedTotalAbsoluteGroupVelocities,
        'totalAbsoluteGroupVelocities': totalAbsoluteGroupVelocities.tolist(),
        'totalAbsoluteGroupVelocityStd': totalAbsoluteGroupVelocityStds.tolist(),
        'vectorialGroupVelocityDifferences': vectorialGroupVelocityDifferences.tolist(),
        'vectorialGroupVelocityDifferencesMean': vectorialGroupVelocityDifferencesMean.tolist(),
        'vectorialGroupVelocityDifferencesStd': vectorialGroupVelocityDifferencesStd.tolist(),
        'totalVectorialVelocity': selectedTotalVectorialVelocity,
        'totalVectorialGroupVelocities': selectedTotalVectorialGroupVelocities,
        'nematicOrderParameters': selectedTotalNematicOrderParameter,
        'totalNematicOrderParameter': totalNematicOrderParameter,
        'totalNematicOrderParameterStd': totalNematicOrderParameterStd,
        'subNematicOrderParametersGroups': selectedTotalNematicOrderParameterGroups,
        'totalNematicOrderParameterGroups': totalNematicOrderParameterGroups.tolist(),
        'totalNematicOrderParameterGroupsStd': totalNematicOrderParameterGroupsStd.tolist(),
        'constants': constants,
        'timeEvolutionVectorialGroupVelocityDifferences': timeEvolutionVectorialGroupVelocityDifferences.tolist(),
        'timeEvolution': timeEvolution.tolist()}


if __name__ == "__main__":
    dataDir = r'/local/kzisiadis/multiple_groups/one_snaking_and_snaking_control_group'
    # dataDir = r'E:\simulationdata\multiple_groups\nematic\sub'
    reevaluateAbsoluteVelocities = False

    # auto find all simulationGroups and evaluate the result for each simulationGroup
    if os.path.exists(dataDir) and os.path.isdir(dataDir):
        resultsPath = os.path.join(dataDir, '_results')
        if not (os.path.exists(resultsPath) and os.path.isdir(resultsPath)):
            os.mkdir(resultsPath)
        else:
            filelist = [f for f in os.listdir(resultsPath) if f.endswith(".json")]
            for file in filelist:
                os.remove(os.path.join(resultsPath, file))

        manager = mp.Manager()
        pool = mp.Pool(processes=(mp.cpu_count() - 2))

        dirList = os.listdir(dataDir)
        simulationGroupPathList = []
        simulationGroups = manager.dict()
        for entry in dirList:
            if os.path.isdir(os.path.join(dataDir, entry)):
                if os.path.exists(os.path.join(dataDir, entry, 'simulationGroupConfig.json')):
                    with open(os.path.join(dataDir, entry, 'simulationGroupConfig.json')) as simulationGroupConfigFile:
                        simulationGroupConfig = json.load(simulationGroupConfigFile)
                # if the config file doesnt exist, count and create a simulationGroupConfig file
                else:
                    simulationDirList = os.listdir(os.path.join(dataDir, entry))
                    numSimulation = 0
                    repeatNum = 0
                    for simulation in simulationDirList:
                        if not os.path.isdir(os.path.join(dataDir, entry, simulation)):
                            continue
                        splitName = simulation.split("_")
                        if len(splitName) != 2:
                            continue

                        simulationNum = int(splitName[0])
                        repeatSimulationNum = int(splitName[1])

                        if simulationNum > numSimulation:
                            numSimulation = simulationNum

                        if repeatSimulationNum > repeatNum:
                            repeatNum = repeatSimulationNum

                    if numSimulation == 0 or repeatNum == 0:
                        continue

                    simulationGroupConfig = {
                        'numSimulation': numSimulation + 1,
                        'repeatNum': repeatNum + 1,
                        'saveTrajectoryData': False,
                        'timePercentageUsedForMean': 25
                    }

                    with open(os.path.join(dataDir, entry, 'simulationGroupConfig.json'),
                              "w") as simulationGroupConfigFile:
                        json.dump(simulationGroupConfig, simulationGroupConfigFile)

                simulationGroups[entry] = {
                    'name': entry,
                    'path': os.path.join(dataDir, entry),
                    'numSimulation': simulationGroupConfig['numSimulation'],
                    'repeatNum': simulationGroupConfig['repeatNum'],
                    'timePercentageUsedForMean': simulationGroupConfig['timePercentageUsedForMean'],
                    'resultDirectory': manager.dict()}

        simulationGroupPool = []
        for simulationGroup in simulationGroups.items():
            simulationGroupData = simulationGroup[1]
            for i in range(simulationGroupData['numSimulation']):
                # simulation pool data:
                # simulationGroupName, simulatinGroupPath, current simulation num, number of repeated simulation of same scenario,
                # timePercentageUsedForMean, simulationGroupDirectory, reevaluateAbsoluteVelocities
                simulationGroupPool.append([
                    simulationGroupData['name'],
                    simulationGroupData['path'],
                    i,
                    simulationGroupData['repeatNum'],
                    simulationGroupData['timePercentageUsedForMean'],
                    simulationGroupData['resultDirectory'],
                    reevaluateAbsoluteVelocities])

        for _ in tqdm(pool.imap(func=calculateResult, iterable=simulationGroupPool),
                      total=len(simulationGroupPool)):
            pass
        pool.close()
        pool.join()

        # after every scenario has been evaluated, retrieve the results and put them in an easy to reach txt
        for simulationGroup in simulationGroups.items():
            simulationGroupData = simulationGroup[1]
            simulationGroupName = simulationGroupData['name']
            sortedDict = sorted(simulationGroupData['resultDirectory'].items(), key=lambda x: x[0])

            environmentSideLengths, groups, randomAngleAmplitude, timeEvolution = [], [], [], []
            timeEvolutionVectorialGroupVelocityDifferences = []
            absoluteVelocities, absoluteVelocity, absoluteVelocityStd = [], [], []
            vectorialGroupVelocityDifferences = []
            vectorialGroupVelocityDifferencesMean, vectorialGroupVelocityDifferencesStd = [], []
            subAbsoluteGroupVelocities = []
            absoluteGroupVelocities, absoluteGroupVelocityStd, vectorialVelocity, vectorialGroupVelocities = [], [], [], []
            nematicOrderParameters, nematicOrderParameter, nematicOrderParamterStd = [], [], []
            subNematicOrderParametersGroups, nematicOrderParameterGroups, nematicOrderParameterGroupsStd = [], [], []

            for entry in sortedDict:
                index = entry[0]
                result = entry[1]

                environmentSideLengths.append(result['constants']['environmentSideLength'])
                groups.append(result['constants']['groups'])
                randomAngleAmplitude.append(result['constants']['randomAngleAmplitude'])
                absoluteVelocities.append(result['absoluteVelocities'])
                absoluteVelocity.append(result['totalAbsoluteVelocity'])
                absoluteVelocityStd.append(result['totalAbsoluteVelocityStd'])
                subAbsoluteGroupVelocities.append(result['subAbsoluteGroupVelocities'])
                absoluteGroupVelocities.append(result['totalAbsoluteGroupVelocities'])
                absoluteGroupVelocityStd.append(result['totalAbsoluteGroupVelocityStd'])
                vectorialGroupVelocityDifferences.append(result['vectorialGroupVelocityDifferences'])
                vectorialGroupVelocityDifferencesMean.append(result['vectorialGroupVelocityDifferencesMean'])
                vectorialGroupVelocityDifferencesStd.append(result['vectorialGroupVelocityDifferencesStd'])
                vectorialVelocity.append(result['totalAbsoluteGroupVelocities'])
                vectorialGroupVelocities.append(result['totalVectorialGroupVelocities'])
                nematicOrderParameters.append(result['nematicOrderParameters'])
                nematicOrderParameter.append(result['totalNematicOrderParameter'])
                nematicOrderParamterStd.append(result['totalNematicOrderParameterStd'])
                subNematicOrderParametersGroups.append(result['subNematicOrderParametersGroups'])
                nematicOrderParameterGroups.append(result['totalNematicOrderParameterGroups'])
                nematicOrderParameterGroupsStd.append(result['totalNematicOrderParameterGroupsStd'])

                timeEvolution.append(result['timeEvolution'])
                timeEvolutionVectorialGroupVelocityDifferences.append(result['timeEvolutionVectorialGroupVelocityDifferences'])


            obj = {
                'environmentSideLengths': environmentSideLengths,
                'groups': groups,
                'randomAngleAmplitude': randomAngleAmplitude,
                'absoluteVelocities': absoluteVelocities,
                'totalAbsoluteVelocity': absoluteVelocity,
                'absoluteVelocityStd': absoluteVelocityStd,
                'subAbsoluteGroupVelocities': subAbsoluteGroupVelocities,
                'absoluteGroupVelocities': absoluteGroupVelocities,
                'absoluteGroupVelocityStd': absoluteGroupVelocityStd,
                'vectorialGroupVelocityDifferences': vectorialGroupVelocityDifferences,
                'vectorialGroupVelocityDifferencesMean': vectorialGroupVelocityDifferencesMean,
                'vectorialGroupVelocityDifferencesStd': vectorialGroupVelocityDifferencesStd,
                'vectorialVelocity': vectorialVelocity,
                'vectorialGroupVelocities': vectorialGroupVelocities,
                'nematicOrderParameters': nematicOrderParameters,
                'nematicOrderParameter': nematicOrderParameter,
                'nematicOrderParameterStd': nematicOrderParamterStd,
                'subNematicOrderParametersGroups': subNematicOrderParametersGroups,
                'nematicOrderParameterGroups': nematicOrderParameterGroups,
                'nematicOrderParameterGroupsStd': nematicOrderParameterGroupsStd,
            }
            # print(obj)

            with open(os.path.join(resultsPath, f'plotResult_{simulationGroupName}.json'), 'w') as plotFile:
                json.dump(obj, plotFile)

            with open(os.path.join(simulationGroupData['path'], f'plotResult_{simulationGroupName}.json'),
                      'w') as plotFile:
                json.dump(obj, plotFile)

            with open(os.path.join(simulationGroupData['path'], f'timeEvolutionResult_{simulationGroupName}.json'),
                      'w') as timeEvolutionFile:
                json.dump(timeEvolution, timeEvolutionFile)

            with open(os.path.join(simulationGroupData['path'], f'timeEvolutionVectorialGroupVelocityDifferencesResult_{simulationGroupName}.json'),
                      'w') as timeEvolutionVectorialGroupVelocityDifferencesFile:
                json.dump(timeEvolutionVectorialGroupVelocityDifferences, timeEvolutionVectorialGroupVelocityDifferencesFile)