import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from scipy.optimize import curve_fit, differential_evolution
from numba import njit
from pprint import pprint


def non_linear_regression(x, y, func, initialParameters):
    # curve fit the test data
    fittedParameters, pcov = curve_fit(func, x, y, initialParameters, maxfev=10 ** 4)

    modelPredictions = func(x, *fittedParameters)

    absError = modelPredictions - y

    SE = np.square(absError)  # squared errors
    MSE = np.mean(SE)  # mean squared errors
    RMSE = np.sqrt(MSE)  # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (np.var(absError) / np.var(y))
    return {'parameters': fittedParameters, 'RMSE': RMSE, 'Rsquared': Rsquared}

@njit
def getMeanAbsolutVelocity(absoluteVelocities, startMean):
    arr = absoluteVelocities[startMean:]
    return np.mean(arr)

if __name__ == "__main__":
    timeEvolutionFile = 'D:/simulationdata/timeEvolutionResult.txt'
    initSimulationScenarioNum = 1

    with open(timeEvolutionFile) as plotFile:
        timeEvolution = json.load(plotFile)

    y = timeEvolution[initSimulationScenarioNum]
    x = np.array(range(len(y)))

    fig, ax = plt.subplots()
    line, = plt.plot(x, y)

    def func(t, a, b):
        return -a * (np.exp(-b * t) - 1)
    initialParameters = [0.5, 0]
    model = non_linear_regression(x / len(y), y, func, initialParameters)
    a, b = model['parameters'][0], model['parameters'][1] / len(y)
    print(a, b)

    xModel = np.linspace(np.min(x), np.max(x), 1000)
    yModel = func(xModel, a, b)
    line2, = plt.plot(xModel, yModel)

    yprimelim = 10**(-5)
    startMean = np.round(np.maximum(1/b * np.log(a * b / yprimelim), 0))
    print(startMean)
    vline = plt.axvline(startMean, 0, 1)
    print("va:", getMeanAbsolutVelocity(np.array(y), startMean))

    plt.subplots_adjust(left=0.25, bottom=0.25)

    axscenarioNum = plt.axes([0.25, 0.1, 0.65, 0.03])
    scenarioSlider = Slider(
        ax=axscenarioNum,
        label='Scenario number',
        valmin=0,
        valmax=len(timeEvolution),
        valinit=initSimulationScenarioNum,
        valstep=1
    )

    def update(newScenarioNum):
        y = timeEvolution[newScenarioNum]
        line.set_ydata(y)
        model = non_linear_regression(x / len(y), y, func, initialParameters)
        a, b = model['parameters'][0], model['parameters'][1] / len(y)
        print(a, b)

        xModel = np.linspace(np.min(x), np.max(x), 1000)
        yModel = func(xModel, a, b)
        line2.set_ydata(yModel)

        startMean = np.round(np.maximum(1/b * np.log(a * b / yprimelim), 0))
        print(startMean)
        vline.set_xdata(startMean)
        print("va:", getMeanAbsolutVelocity(np.array(y), startMean))

    scenarioSlider.on_changed(update)

    plt.ylim([0, 1.1])
    plt.show()
    plt.close('all')
