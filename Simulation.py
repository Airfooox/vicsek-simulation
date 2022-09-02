import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform
from numba import njit

import util
from util import printProgressBar
import time


class Simulation:
    def __init__(self, simulationIndex, numSimulation, CONSTANTS, initialParameterFunc):
        self.simulationConstants = CONSTANTS
        self.timeSteps = self.simulationConstants["timeSteps"]
        self.framesUsedForMean = np.ceil(((1 - (self.simulationConstants["timePercentageUsedForMean"] / 100)) * self.timeSteps))

        # initialize the swimmers with position, angle of velocity, amplitude, period and phase difference
        self.swimmers = []
        # self.swimmers.append([0, 00, 0, np.pi / 20, 60, 0])
        # self.swimmers.append([0, -1, 0, np.pi / 20, 60, np.pi])
        for swimmerIndex in range(self.simulationConstants["numSwimmers"]):
            initialParameter = initialParameterFunc(simulationIndex, numSimulation, swimmerIndex)

            xPos = - self.simulationConstants["environmentSideLength"] / 4  #= np.random.rand() * self.simulationConstants["environmentSideLength"] - self.simulationConstants["environmentSideLength"] / 2
            yPos = 0 #= np.random.rand() * self.simulationConstants["environmentSideLength"] - self.simulationConstants["environmentSideLength"] / 2
            vPhi = np.random.rand() * 2 * np.pi * 0 # angle in rad
            oscillationAmplitude = initialParameter["oscillationAmplitude"]
            oscillationPeriod = initialParameter["oscillationPeriod"]
            oscillationPhaseShift = initialParameter["oscillationPhaseshift"]
            self.swimmers.append([xPos, yPos, vPhi, oscillationAmplitude, oscillationPeriod, oscillationPhaseShift])

        # initialize states array and add initial values for first frame
        self.states = np.empty((self.timeSteps, self.simulationConstants["numSwimmers"], 6), dtype=np.float64)
        for swimmerIndex in range(self.simulationConstants["numSwimmers"]):
            swimmerState = np.array([self.swimmers[swimmerIndex]], dtype=np.float64)
            self.states[0][swimmerIndex] = swimmerState

        self.absoluteVelocities = np.empty((self.timeSteps), dtype=np.float64)

    def simulate(self):
        for t in range (1, self.timeSteps):
           self.states[t], self.absoluteVelocities[t]  = self.simulationStep(t)

            # printProgressBar(t, self.timeSteps - 1, prefix='Simulation Progress:', suffix='Simulation Complete', length=50)

        return self.states

    # @njit
    def simulationStep(self, t):
        previousState = []
        try:
            previousState = self.states[t - 1]
        except IndexError:
            previousState = self.simulationStep(t - 1)



        # _pdist = pdist(previousState[:, :2])
        distances = squareform(pdist(previousState[:, :2])) # symmetric matric with all distances between the particles
        index1, index2 = np.where(distances <= self.simulationConstants["interactionRadius"])
        uniqueDistance = (index1 != index2) # because the matrix is symmetric, throw out all diagonal, because its all zero
        index1 = index1[uniqueDistance]
        index2 = index2[uniqueDistance]

        newState = previousState.copy()

        numSwimmers = self.simulationConstants["numSwimmers"]
        environmentSideLength = self.simulationConstants["environmentSideLength"]
        interactionRadius = self.simulationConstants["interactionRadius"]
        randomAngleAmplitude = self.simulationConstants["randomAngleAmplitude"]
        initialVelocity = self.simulationConstants["initialVelocity"]

        # [newState, absoluteVelocity] =
        return self.calculateNewState(t, previousState, newState, index1, index2, numSwimmers, environmentSideLength, interactionRadius, randomAngleAmplitude, initialVelocity)

    def getStates(self):
        return self.states

    def getAbsoluteVelocities(self):
        return self.absoluteVelocities

    def getAbsoluteVelocityTotal(self):
        return Simulation.calculateAbsoluteVelocityTotal(self.timeSteps, self.framesUsedForMean, self.absoluteVelocities)

    @staticmethod
    @njit
    def calculateNewState(t, previousState, newState, index1, index2, numSwimmers, environmentSideLength, interactionRadius, randomAngleAmplitude, initialVelocity):
        sumVelocity = np.array([0, 0], dtype=np.float64)
        for i in range(numSwimmers):
            swimmerState = newState[i]

            # interaction over boundary stuff
            # check if particles radius is over boundary
            # -> shadow teleport that particle over the boundary on the other side
            # get distances to all particles
            # (make it smart, so if in right upper corner, teleport to lower left)
            leftInteractionBoundaryHit = (swimmerState[0] < -environmentSideLength / 2 +
                                          interactionRadius)
            rightInteractionBoundaryHit = (swimmerState[0] > environmentSideLength / 2 -
                                           interactionRadius)
            upInteractionBoundaryHit = (swimmerState[1] > environmentSideLength / 2 -
                                        interactionRadius)
            downInteractionBoundaryHit = (swimmerState[1] < -environmentSideLength / 2 +
                                          interactionRadius)

            shadowSwimmerPositions = np.empty((0), dtype=np.float64)
            if leftInteractionBoundaryHit:
                newShadowState = swimmerState.copy()
                newShadowState[0] += environmentSideLength
                shadowSwimmerPositions = np.append(shadowSwimmerPositions, newShadowState)

                if upInteractionBoundaryHit:
                    newShadowStateUp = swimmerState.copy()
                    newShadowStateUp[0] += environmentSideLength
                    newShadowStateUp[1] -= environmentSideLength
                    shadowSwimmerPositions = np.append(shadowSwimmerPositions, newShadowStateUp)

                if downInteractionBoundaryHit:
                    newShadowStateDown = swimmerState.copy()
                    newShadowStateDown[0] += environmentSideLength
                    newShadowStateDown[1] += environmentSideLength
                    shadowSwimmerPositions = np.append(shadowSwimmerPositions, newShadowStateDown)
            if rightInteractionBoundaryHit:
                newShadowState = swimmerState.copy()
                newShadowState[0] -= environmentSideLength
                shadowSwimmerPositions = np.append(shadowSwimmerPositions, newShadowState)

                if upInteractionBoundaryHit:
                    newShadowStateUp = swimmerState.copy()
                    newShadowStateUp[0] -= environmentSideLength
                    newShadowStateUp[1] -= environmentSideLength
                    shadowSwimmerPositions = np.append(shadowSwimmerPositions, newShadowStateUp)

                if downInteractionBoundaryHit:
                    newShadowStateDown = swimmerState.copy()
                    newShadowStateDown[0] -= environmentSideLength
                    newShadowStateDown[1] += environmentSideLength
                    shadowSwimmerPositions = np.append(shadowSwimmerPositions, newShadowStateDown)
            if upInteractionBoundaryHit:
                newShadowState = swimmerState.copy()
                newShadowState[1] -= environmentSideLength
                shadowSwimmerPositions = np.append(shadowSwimmerPositions, newShadowState)
            if downInteractionBoundaryHit:
                newShadowState = swimmerState.copy()
                newShadowState[1] += environmentSideLength
                shadowSwimmerPositions = np.append(shadowSwimmerPositions, newShadowState)

            # shadowSwimmerPositions is up to this point a 1-D array with all the positions and velocity angles like [x1, y1, phi1, x2, y2, phi2, ..., xn, yn, phin]
            # reshape it into a 2-D array: [[x1, y1, phi1], [x2, y2, phi2], ..., [xn, yn, phin]]
            shadowSwimmerPositions = np.reshape(shadowSwimmerPositions, (-1, 3))
            numberOfShadowPositions = len(shadowSwimmerPositions)
            if numberOfShadowPositions > 0:
                for j in range(numSwimmers):
                    if i == j:
                        continue

                    for k in range(numberOfShadowPositions):
                        shadowPosition = shadowSwimmerPositions[k]
                        shadowDistanceSwimmer = previousState[j]
                        distanceBetween = (shadowPosition[0] - shadowDistanceSwimmer[0]) ** 2 + (
                                    shadowPosition[1] - shadowDistanceSwimmer[1]) ** 2
                        if distanceBetween ** 2 <= interactionRadius:
                            index1 = np.append(index1, [i])
                            index2 = np.append(index2, [j])

            indexForDistance = index2[index1 == i]
            # i and j are in range
            sinSum = np.sin(swimmerState[2])
            cosSum = np.cos(swimmerState[2])
            for j in indexForDistance:
                swimmerInRange = previousState[j]
                # distance = distances[i, j]
                sinSum += np.sin(swimmerInRange[2])
                cosSum += np.cos(swimmerInRange[2])

            averageAngle = np.arctan2(sinSum, cosSum) * 0
            randomAngle = ((np.random.rand() - 0.5) * randomAngleAmplitude)
            cosinusOscillation = swimmerState[3] * np.cos(
                (2 * np.pi / swimmerState[4]) * t + swimmerState[5])
            swimmerState[2] = averageAngle + randomAngle + cosinusOscillation

            xVelCos = np.cos(swimmerState[2])
            yVelSin = np.sin(swimmerState[2])
            sumVelocity += np.array([xVelCos, yVelSin], dtype=np.float64)

            swimmerState[0] += initialVelocity * xVelCos
            swimmerState[1] += initialVelocity * yVelSin

            # check for boundary hits
            leftBoundaryHit = (swimmerState[0] <= -environmentSideLength / 2)
            rightBoundaryHit = (swimmerState[0] >= environmentSideLength / 2)
            upBoundaryHit = (swimmerState[1] >= environmentSideLength / 2)
            downBoundaryHit = (swimmerState[1] <= -environmentSideLength / 2)

            if leftBoundaryHit:
                swimmerState[0] += environmentSideLength
            if rightBoundaryHit:
                swimmerState[0] += -environmentSideLength
            if upBoundaryHit:
                swimmerState[1] += -environmentSideLength
            if downBoundaryHit:
                swimmerState[1] += environmentSideLength

        return newState, np.linalg.norm(sumVelocity) / numSwimmers

    @staticmethod
    @njit
    def getMeanAbsolutVelocity(absoluteVelocities, startMean):
        arr = absoluteVelocities[startMean:]
        mean = 0
        for absoluteVelocity in arr:
            mean += absoluteVelocity
        return mean / len(arr)

    @staticmethod
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

    @staticmethod
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
            model = Simulation.nonLinearFitting(t / len(absoluteVelocities), absoluteVelocities, func, initialParameters)
            a, b = model['parameters'][0], model['parameters'][1] / len(absoluteVelocities)

            # find the time when system is in saturation for getting the mean value of absolut velocities
            yprimelim = 10 ** (-5)
            saturationBorder = np.round(np.maximum(1/b * np.log(a * b / yprimelim), 0))
            if saturationBorder > framesUsedForMean:
                return -3
        except RuntimeError:
            return -1
        try:
            absolutVelocity = Simulation.getMeanAbsolutVelocity(np.array(absoluteVelocities), framesUsedForMean)
        except ZeroDivisionError:
            absolutVelocity = -2
            # print(timeSteps, np.array(absoluteVelocities))
        return absolutVelocity

    def animate(self):
        # initialize animation
        self.figure = plt.figure()
        self.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        limits = self.simulationConstants["environmentSideLength"] / 2 + 0.2
        self.axis = self.figure.add_subplot(111, aspect='equal', autoscale_on=False,
                             xlim=(-limits, limits), ylim=(-limits, limits))

        self.rect = plt.Rectangle((-self.simulationConstants["environmentSideLength"] / 2, -self.simulationConstants["environmentSideLength"] / 2),
                             self.simulationConstants["environmentSideLength"],
                             self.simulationConstants["environmentSideLength"],
                             ec='none', lw=2, fc='none')
        self.axis.add_patch(self.rect)

        # initialize plot with empty position data, cyclic colormap and swimmer phase as data for the color map
        initialState = self.states[0]
        initialX = initialState[:, 0]
        initialY = initialState[:, 1]
        initialPhi = initialState[:, 2]
        self.swimmerPlot = self.axis.quiver(initialX, initialY, np.cos(initialPhi), np.sin(initialPhi), initialPhi % 2*np.pi, cmap=plt.get_cmap("twilight"), clim=[0, 2*np.pi], pivot="middle", scale=250, units="width", width=0.0005, headwidth= 30, headlength=50, headaxislength=45,minshaft=0.99, minlength=0)
        self.trajectoryLines = []
        for i in range(self.simulationConstants["numSwimmers"]):
            trajectoryLine, = self.axis.plot([], [], "go", ms=0.75)
            self.trajectoryLines.append(trajectoryLine)

        self.figure.colorbar(self.swimmerPlot, ax=self.axis)

        def plotInit():
            # microswimmersPlot.set_offsets([], [])
            self.rect.set_edgecolor('none')
            for i in range(self.simulationConstants["numSwimmers"]):
                self.trajectoryLines[i].set_data([], [])

            # an array of all updates objects must be returned in this function, here the scatter plot, rect and all of the trajectory line plots
            iterableArtists = [self.swimmerPlot, self.rect] + self.trajectoryLines
            return iterableArtists

        def animate(t):
            data = self.states[t]

            markerSize = int(self.figure.dpi * 2 * self.simulationConstants["swimmerSize"] * self.figure.get_figwidth()
                             / np.diff(self.axis.get_xbound())[0])
            areaSize = np.ones(self.simulationConstants["numSwimmers"]) * 0.5 * markerSize ** 2

            self.rect.set_edgecolor('k')

            # update pieces of the animation
            x = data[:, 0]
            y = data[:, 1]
            phi = data[:, 2]
            phase = data[:, 5]
            # combine x and y data of objects in following way:
            # [x1, x2, ..., xn] and [y1, y2, ..., yn] -> [[x1, y1], [x2,y2], ..., [xn, yn]]
            self.swimmerPlot.set_offsets(np.c_[x, y])
            self.swimmerPlot.set_sizes(areaSize)
            self.swimmerPlot.set_UVC(np.cos(phi), np.sin(phi) , phi % 2*np.pi)

            for i in range(self.simulationConstants["numSwimmers"]):
                self.trajectoryLines[i].set_data(self.states[:t, i, 0], self.states[:t, i, 1])

            # ax.text(0.5, 0.5, "Zeit: {} s".format(i / CONSTANTS["frames"]))

            # an array of all updates objects must be returned in this function, here the scatter plot, rect and all of the trajectory line plots
            iterableArtists = [self.swimmerPlot, self.rect] + self.trajectoryLines
            return iterableArtists

        ani = animation.FuncAnimation(self.figure, func=animate, frames=self.timeSteps,
                                      interval=1000 / 60, blit=True, init_func=plotInit)

        # mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\konst\\Desktop\\ffmpeg\\bin\\ffmpeg.exe'
        # f = r"c:\\Users\\konst\\Desktop\\animation.mp4"
        # writervideo = animation.FFMpegWriter(fps=self.simulationConstants["fps"], bitrate=3000)
        # ani.save(
        #     f, writer=writervideo,
        #     progress_callback=lambda i, n: printProgressBar(i, n, prefix='Animation Progress:',
        #                                                     suffix='Animation Complete', length=50)
        # )

        plt.show()

