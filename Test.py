import numpy as np
import json
from numba import njit
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

if __name__ == "__main__":
    dir = 'D:/simulationdata/sameRhoGroup400PhaseShift90_23_84'

    with open(dir + '/constants.txt') as constantFile:
        constants = json.load(constantFile)
    timeEvolutionData = np.load(dir + '/absoluteVelocities.npy')
    if os.path.exists(dir + '/statesData.npy'):
        states = np.load(dir + '/statesData.npy')

    # with open(dir + '/timeEvolution.txt', 'w') as timeEvolutionFile:
    #     json.dump(timeEvolutionData.tolist(), timeEvolutionFile)

    @njit
    def getMeanAbsolutVelocity(absoluteVelocities, framesUsedForMean):
        arr = absoluteVelocities[framesUsedForMean:]
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

    def animate():
        # initialize animation
        figure = plt.figure()
        figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        limits = constants["environmentSideLength"] / 2 + 0.2
        axis = figure.add_subplot(111, aspect='equal', autoscale_on=False,
                             xlim=(-limits, limits), ylim=(-limits, limits))

        rect = plt.Rectangle((-constants["environmentSideLength"] / 2, -constants["environmentSideLength"] / 2),
                             constants["environmentSideLength"],
                             constants["environmentSideLength"],
                             ec='none', lw=2, fc='none')
        axis.add_patch(rect)

        swimmerPlot = axis.scatter([], [])
        swimmerColors = np.random.rand(constants["numSwimmers"])
        timeSteps = constants["timeSteps"]

        def plotInit():
            # microswimmersPlot.set_offsets([], [])
            rect.set_edgecolor('none')
            return swimmerPlot, rect

        def animate(i):
            data = states[i]

            markerSize = int(figure.dpi * 2 * constants["swimmerSize"] * figure.get_figwidth()
                             / np.diff(axis.get_xbound())[0])
            areaSize = np.ones(constants["numSwimmers"]) * 0.5 * markerSize ** 2

            # update pieces of the animation
            rect.set_edgecolor('k')

            x = data[:, 0]
            y = data[:, 1]
            # combine x and y data of objects in following way:
            # [x1, x2, ..., xn] and [y1, y2, ..., yn] -> [[x1, y1], [x2,y2], ..., [xn, yn]]
            swimmerPlot.set_offsets(np.c_[x, y])
            swimmerPlot.set_sizes(areaSize)
            swimmerPlot.set_array(swimmerColors)

            # ax.text(0.5, 0.5, "Zeit: {} s".format(i / CONSTANTS["frames"]))
            return swimmerPlot, rect

        ani = animation.FuncAnimation(figure, animate, frames=timeSteps,
                                      interval=1000 / 30, blit=True, init_func=plotInit)

        # mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\konst\\Desktop\\ffmpeg\\bin\\ffmpeg.exe'
        # f = r"c:\\Users\\konst\\Desktop\\animation.mp4"
        # writervideo = animation.FFMpegWriter(fps=constants["fps"], bitrate=3000)
        # ani.save(
        #     f, writer=writervideo,
        #     progress_callback=lambda i, n: printProgressBar(i, n, prefix='Animation Progress:',
        #                                                     suffix='Animation Complete', length=50)
        # )

        plt.show()

    def calculateAbsoluteVelocityTotal(timeSteps, framesUsedForMean, absoluteVelocities):
        absoluteVelocities = np.array(absoluteVelocities)
        absoluteVelocities[np.abs(absoluteVelocities) < np.finfo(float).eps] = 0

        # defining exponential function for saturation
        def func(t, a, b):
            return -a * (np.exp(-b * t) - 1)

        # find parameters for saturation function
        t = np.array(range(timeSteps))
        initialParameters = [0.5, 0.1]

        model = nonLinearFitting(t / len(absoluteVelocities), absoluteVelocities, func, initialParameters)
        a, b = model['parameters'][0], model['parameters'][1] / len(absoluteVelocities)

        # find the time when system is in saturation for getting the mean value of absolut velocities
        yprimelim = 10 ** (-5)
        saturationBorder = np.round(np.maximum(1 / b * np.log(a * b / yprimelim), 0))

        fig, ax = plt.subplots()
        plt.plot(t, absoluteVelocities)
        xModel = np.linspace(0, t, 1000)
        print(a, b)
        yModel = func(xModel, a, b)
        plt.plot(xModel, yModel)
        plt.axvline(saturationBorder, 0, 1, color="g")
        plt.axvline(framesUsedForMean, 0, 1, color="r")

        plt.ylim([0, 1.1])

        try:
            if saturationBorder > framesUsedForMean:
                print(-3)
        except RuntimeError:
            print(-1)
        try:
            absolutVelocity = getMeanAbsolutVelocity(np.array(absoluteVelocities), framesUsedForMean)
        except ZeroDivisionError:
            absolutVelocity = -2
            print(timeSteps, np.array(absoluteVelocities))

        plt.show()
        plt.close('all')
        return absolutVelocity


    framesUsedForMean = np.ceil(((1 - (constants["timePercentageUsedForMean"] / 100)) * constants["timeSteps"]))
    print(calculateAbsoluteVelocityTotal(len(timeEvolutionData), framesUsedForMean, timeEvolutionData))
    # animate()