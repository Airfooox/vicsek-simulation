import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.widgets import Slider
from scipy.optimize import curve_fit, differential_evolution
from numba import njit
from pprint import pprint

import util

mpl.use("TkAgg")


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
    # timeEvolutionFile = r'E:\simulationdata\simulations\sameRhoGroup_phaseShift_N=1000_eta=2_A=0.06pi_T=30_phi=[0.0pi]_g=1\timeEvolutionResult_sameRhoGroup_phaseShift_N=1000_eta=2_A=0.06pi_T=30_phi=[0.0pi]_g=1.txt'
    # timeEvolutionFile = r'C:\Users\konst\OneDrive\Uni\Lehre\7. Semester\Bachelorarbeit\simulationData\simulationData\multiple_groups\pi_over_8_and_no_snaking\timeEvolutionResult_va_over_eta_multiple_groups_d5eae441-9e31-459b-8b35-10fb39de70dc.json'
    timeEvolutionFile = r'C:\Users\konst\OneDrive\Uni\Lehre\7. Semester\Bachelorarbeit\simulationData\simulationData\multiple_groups\timeEvolutionVectorialGroupVelocityDifferencesResult_va_over_eta_multiple_groups_one_snaking_other_not.json'
    initSimulationScenarioNum = 1

    timePercentageUsedForMean = 25

    with open(timeEvolutionFile) as plotFile:
        timeEvolution = json.load(plotFile)

    y = np.array(timeEvolution)[initSimulationScenarioNum, :, 0]
    x = np.array(range(len(y)))

    framesUsedForMean = int(np.ceil(((1 - (timePercentageUsedForMean / 100)) * len(y))))

    fig, ax = plt.subplots()
    ax.set_ylim(-0.05, np.pi + 0.05)
    line, = plt.plot(x, y, marker='x', linestyle='', markersize=1)
    mean = np.mean(y[framesUsedForMean:])
    std = np.std(y[framesUsedForMean:])
    meanLine = plt.axhline(y=mean, color='r', linestyle='-')
    meanOStdLine = plt.axhline(y=mean+std, color='r', linestyle='--')
    meanUStdLine = plt.axhline(y=mean-std, color='r', linestyle='--')
    print('mean/std:', mean, std)

    majorTicks = util.Multiple(8)
    ax.yaxis.set_major_locator(majorTicks.locator())
    ax.yaxis.set_major_formatter(majorTicks.formatter())

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
    saturationBorder = np.round(np.maximum(1/b * np.log(a * b / yprimelim), 0))
    print(saturationBorder)
    vline = plt.axvline(saturationBorder, 0, 1, color="g")
    plt.axvline(framesUsedForMean, 0, 1, color="r")
    print("va:", getMeanAbsolutVelocity(np.array(y), saturationBorder))

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
        y = np.array(timeEvolution)[newScenarioNum, :, 0]
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

        mean = np.mean(y[framesUsedForMean:])
        std = np.std(y[framesUsedForMean:])
        meanLine.set_ydata(mean)
        meanOStdLine.set_ydata(mean + std)
        meanUStdLine.set_ydata(mean - std)
        print('mean/std:', mean, std)

    scenarioSlider.on_changed(update)

    plt.ylim([0, 1.1])
    plt.show()
    plt.close('all')
