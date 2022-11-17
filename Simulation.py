from __future__ import annotations

import numba.core.types
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import curve_fit
from numba import uint16, float32, types, njit
from numba.experimental import jitclass
from util import printProgressBar


@jitclass([
    ('width', float32),
    ('height', float32),
    ('cellSize', float32),
    ('columns', uint16),
    ('rows', uint16),
    ('cells', types.List(types.Array(uint16, 1, 'C')))
])
class Grid:
    def __init__(self, width, height, cellSize):
        self.width = np.float32(width)
        self.height = np.float32(height)
        self.cellSize = np.float32(cellSize)

        # ceil division
        #https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python/17511341#17511341
        self.columns = np.uint16(-(self.width // -self.cellSize))
        self.rows = np.uint16(-(self.height // -self.cellSize))

        self.cells = [np.zeros(0, dtype=np.uint16) for _ in range(self.rows * self.columns)]

    def addSwimmer(self, cellIndex: int, swimmerId: int) -> None:
        self.cells[cellIndex] = np.append(self.cells[cellIndex], np.uint16(swimmerId))
        # self.cells[cellIndex].append(np.uint16(swimmerId))

    def removeSwimmer(self, cellIndex: int, swimmerId: int) -> None:
        cell = self.cells[cellIndex]
        self.cells[cellIndex] = cell[cell != np.uint16(swimmerId)]

    def getNeighbourCellsSwimmerIndices(self, x: float, y: float):
        swimmerIndices = np.zeros(0, dtype=np.uint16)

        minColumn = int((x - self.cellSize) // self.cellSize)
        maxColumn = int((x + self.cellSize) // self.cellSize)
        minRow = int((y - self.cellSize) // self.cellSize)
        maxRow = int((y + self.cellSize) // self.cellSize)

        # num of columns/rows is self.columns and self.rows, but the indices start from 0,
        # so we have to subtract 1 from the max num of column/row to get the index of the max column/row
        if minColumn < 0:
            minColumn = 0
        if maxColumn > self.columns - 1:
            maxColumn = self.columns - 1
        if minRow < 0:
            minRow = 0
        if maxRow > self.rows - 1:
            maxRow = self.rows - 1

        # +1 because range doesn't include the upper limit
        for row in range(minRow, maxRow + 1):
            for column in range(minColumn, maxColumn + 1):
                # print('row column', row, column, 'cell', column + self.rows * row)
                # print('cell swimmers', self.cells[column + self.rows * row])
                swimmerIndices = np.append(swimmerIndices, self.cells[column + self.rows * row])
        return swimmerIndices

    def getCellIndex(self, x: float, y: float):
        return int(x // self.cellSize) + int(y // self.cellSize) * self.rows


class Simulation:
    def __init__(self, simulationIndex, numSimulation, constants, timePercentageUsedForMean):
        self.simulationConstants = constants
        self.framesUsedForMean = np.ceil(
            ((1 - (timePercentageUsedForMean / 100)) * self.simulationConstants["timeSteps"]))

        self.grid = Grid(self.simulationConstants['environmentSideLength'], self.simulationConstants['environmentSideLength'], self.simulationConstants['interactionRadius'])

        # initialize the swimmers with position, angle of velocity, amplitude, period and phase shift
        initialSwimmerState = []
        for groupIdentifier, groupConfig in self.simulationConstants['groups'].items():
            for swimmerIndex in range(groupConfig["numSwimmers"]):
                xPos = np.random.rand() * self.simulationConstants["environmentSideLength"]
                yPos = np.random.rand() * self.simulationConstants["environmentSideLength"]
                vPhi = np.random.rand() * 2 * np.pi  # angle in rad
                oscillationAmplitude = groupConfig["oscillationAmplitude"]
                oscillationPeriod = groupConfig["oscillationPeriod"]
                oscillationPhaseShift = groupConfig["oscillationPhaseshift"]
                initialSwimmerState.append(
                    [xPos, yPos, vPhi, oscillationAmplitude, oscillationPeriod, oscillationPhaseShift])

                self.grid.addSwimmer(self.grid.getCellIndex(xPos, yPos), swimmerIndex)
        self.numSwimmers = len(initialSwimmerState)

        # initialize states array and add initial values for first frame
        self.states = np.zeros((self.simulationConstants["timeSteps"], self.numSwimmers, 6), dtype=np.float64)
        self.absoluteVelocities = np.zeros((self.simulationConstants["timeSteps"]), dtype=np.float64)

        self.states[0] = np.array(initialSwimmerState, dtype=np.float64)

    def simulate(self):
        for t in range(1, self.simulationConstants["timeSteps"]):
            self.grid, self.states[t], self.absoluteVelocities[t] = self.simulationStep(t)

            # printProgressBar(t, self.simulationConstants["timeSteps"] - 1, prefix='Simulation Progress:', suffix='Simulation Complete', length=50)

        return self.states

    # @jit
    def simulationStep(self, t):
        previousState = []
        try:
            previousState = self.states[t - 1]
        except IndexError:
            previousState = self.simulationStep(t - 1)

        environmentSideLength = self.simulationConstants['environmentSideLength']
        randomAngleAmplitude = self.simulationConstants['randomAngleAmplitude']
        interactionRadius = self.simulationConstants['interactionRadius']
        initialVelocity = self.simulationConstants['initialVelocity']

        return self.calculateNewState(t, self.grid, previousState, self.numSwimmers, environmentSideLength, randomAngleAmplitude, interactionRadius, initialVelocity)

    def getAbsoluteVelocityTotal(self):
        return Simulation.calculateAbsoluteVelocityTotal(self.simulationConstants["timeSteps"], self.framesUsedForMean,
                                                         self.absoluteVelocities)

    @staticmethod
    @njit
    def calculateNewState(t, grid, previousState, numSwimmers, environmentSideLength, randomAngleAmplitude, interactionRadius, initialVelocity):
        sumVelocity = np.array([0, 0], dtype=np.float64)
        newState = previousState.copy()
        for swimmerIndex in range(numSwimmers):
        # for swimmerIndex in range(1):
            swimmerState = newState[swimmerIndex]
            interactionSwimmerIndices = np.zeros(0, dtype=np.uint16)

            # potential interaction swimmer indices
            gridInteractionSwimmerIndices = grid.getNeighbourCellsSwimmerIndices(swimmerState[0], swimmerState[1])
            # print('SWIMMER', swimmerState[0], swimmerState[1], grid.getCellIndex(swimmerState[0], swimmerState[1]), gridInteractionSwimmerIndices)
            for interactionSwimmerIndex in gridInteractionSwimmerIndices:
                # no self interaction, because we already take the current state as the base values
                if interactionSwimmerIndex == swimmerIndex:
                    continue

                interactionSwimmerState = previousState[interactionSwimmerIndex]
                distanceBetween = (swimmerState[0] - interactionSwimmerState[0]) ** 2 + (swimmerState[1] - interactionSwimmerState[1]) ** 2
                if distanceBetween <= interactionRadius ** 2:
                    interactionSwimmerIndices = np.append(interactionSwimmerIndices, interactionSwimmerIndex)
            # print(interactionSwimmerIndices)

            # interaction over boundary stuff
            # check if particles radius is over boundary
            # -> shadow teleport that particle over the boundary on the other side
            # get distances to all particles
            # (make it smart, so if in right upper corner, teleport to lower left)
            leftInteractionBoundaryHit = (swimmerState[0] < interactionRadius)
            rightInteractionBoundaryHit = (swimmerState[0] > environmentSideLength - interactionRadius)
            upInteractionBoundaryHit = (swimmerState[1] > environmentSideLength - interactionRadius)
            downInteractionBoundaryHit = (swimmerState[1] < interactionRadius)

            shadowSwimmerPositions = np.zeros((0), dtype=np.float64)
            if leftInteractionBoundaryHit:
                newShadowState = swimmerState.copy()
                newShadowState[0] += environmentSideLength
                shadowSwimmerPositions = np.append(shadowSwimmerPositions, newShadowState[:2])

                if upInteractionBoundaryHit:
                    newShadowStateUp = swimmerState.copy()
                    newShadowStateUp[0] += environmentSideLength
                    newShadowStateUp[1] -= environmentSideLength
                    shadowSwimmerPositions = np.append(shadowSwimmerPositions, newShadowStateUp[:2])

                if downInteractionBoundaryHit:
                    newShadowStateDown = swimmerState.copy()
                    newShadowStateDown[0] += environmentSideLength
                    newShadowStateDown[1] += environmentSideLength
                    shadowSwimmerPositions = np.append(shadowSwimmerPositions, newShadowStateDown[:2])
            if rightInteractionBoundaryHit:
                newShadowState = swimmerState.copy()
                newShadowState[0] -= environmentSideLength
                shadowSwimmerPositions = np.append(shadowSwimmerPositions, newShadowState[:2])

                if upInteractionBoundaryHit:
                    newShadowStateUp = swimmerState.copy()
                    newShadowStateUp[0] -= environmentSideLength
                    newShadowStateUp[1] -= environmentSideLength
                    shadowSwimmerPositions = np.append(shadowSwimmerPositions, newShadowStateUp[:2])

                if downInteractionBoundaryHit:
                    newShadowStateDown = swimmerState.copy()
                    newShadowStateDown[0] -= environmentSideLength
                    newShadowStateDown[1] += environmentSideLength
                    shadowSwimmerPositions = np.append(shadowSwimmerPositions, newShadowStateDown[:2])
            if upInteractionBoundaryHit:
                newShadowState = swimmerState.copy()
                newShadowState[1] -= environmentSideLength
                shadowSwimmerPositions = np.append(shadowSwimmerPositions, newShadowState[:2])
            if downInteractionBoundaryHit:
                newShadowState = swimmerState.copy()
                newShadowState[1] += environmentSideLength
                shadowSwimmerPositions = np.append(shadowSwimmerPositions, newShadowState[:2])

            # shadowSwimmerPositions is up to this point a 1-D array with all the positions and velocity angles like [x1, y1, x2, y2, ..., xn, yn]
            # reshape it into a 2-D array: [[x1, y1], [x2, y2], ..., [xn, yn]]
            shadowSwimmerPositions = np.reshape(shadowSwimmerPositions, (-1, 2))
            for shadowSwimmerPosition in shadowSwimmerPositions:
                # get the possible interactions from the shadow swimmers
                gridShadowInteractionSwimmerIndices = grid.getNeighbourCellsSwimmerIndices(shadowSwimmerPosition[0], shadowSwimmerPosition[1])
                # print('SHADOW SWIMMER', shadowSwimmerPosition[0], shadowSwimmerPosition[1], gridShadowInteractionSwimmerIndices)
                for shadowInteractionSwimmerIndex in gridShadowInteractionSwimmerIndices:
                    if shadowInteractionSwimmerIndex == swimmerIndex:
                        continue

                    if shadowInteractionSwimmerIndex in interactionSwimmerIndices:
                        continue

                    shadowInteractionSwimmerState = previousState[shadowInteractionSwimmerIndex]
                    distanceBetween = (shadowSwimmerPosition[0] - shadowInteractionSwimmerState[0]) ** 2 + (
                                shadowSwimmerPosition[1] - shadowInteractionSwimmerState[1]) ** 2
                    if distanceBetween ** 2 <= interactionRadius:
                        interactionSwimmerIndices = np.append(interactionSwimmerIndices, shadowInteractionSwimmerIndex)
            # print(interactionSwimmerIndices)

            # i and j are in range
            sinSum = np.sin(swimmerState[2])
            cosSum = np.cos(swimmerState[2])
            for interactionSwimmerIndex in interactionSwimmerIndices:
                interactionSwimmerState = previousState[interactionSwimmerIndex]
                sinSum += np.sin(interactionSwimmerState[2])
                cosSum += np.cos(interactionSwimmerState[2])

            averageAngle = np.arctan2(sinSum, cosSum)
            randomAngle = ((np.random.rand() - 0.5) * randomAngleAmplitude)
            cosinesOscillation = swimmerState[3] * np.cos(
                (2 * np.pi / swimmerState[4]) * t + swimmerState[5])
            swimmerState[2] = averageAngle + randomAngle + cosinesOscillation

            xVelCos = np.cos(swimmerState[2])
            yVelSin = np.sin(swimmerState[2])
            sumVelocity += np.array([xVelCos, yVelSin], dtype=np.float64)

            swimmerState[0] += initialVelocity * xVelCos
            swimmerState[1] += initialVelocity * yVelSin

            # check for boundary hits
            leftBoundaryHit = (swimmerState[0] <= 0)
            rightBoundaryHit = (swimmerState[0] >= environmentSideLength)
            upBoundaryHit = (swimmerState[1] >= environmentSideLength)
            downBoundaryHit = (swimmerState[1] <= 0)

            if leftBoundaryHit:
                swimmerState[0] += environmentSideLength
            if rightBoundaryHit:
                swimmerState[0] += -environmentSideLength
            if upBoundaryHit:
                swimmerState[1] += -environmentSideLength
            if downBoundaryHit:
                swimmerState[1] += environmentSideLength

            previousSwimmerGridIndex = grid.getCellIndex(previousState[swimmerIndex][0], previousState[swimmerIndex][1])
            newSwimmerGridIndex = grid.getCellIndex(swimmerState[0], swimmerState[1])
            if previousSwimmerGridIndex != newSwimmerGridIndex:
                grid.removeSwimmer(previousSwimmerGridIndex, swimmerIndex)
                grid.addSwimmer(newSwimmerGridIndex, swimmerIndex)

        return grid, newState, np.linalg.norm(sumVelocity) / numSwimmers

    @staticmethod
    @njit
    def getMeanAbsolutVelocity(absoluteVelocities, startMean):
        return np.mean(absoluteVelocities[startMean:])

    @staticmethod
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
            model = Simulation.nonLinearFitting(t / len(absoluteVelocities), absoluteVelocities, func,
                                                initialParameters)
            a, b = model['parameters'][0], model['parameters'][1] / len(absoluteVelocities)

            # find the time when system is in saturation for getting the mean value of absolut velocities
            yprimelim = 10 ** (-5)
            saturationBorder = np.round(np.maximum(1 / b * np.log(a * b / yprimelim), 0))
            if saturationBorder > framesUsedForMean:
                pass
                # return -3
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
        limits = self.simulationConstants["environmentSideLength"] + 0.2
        self.axis = self.figure.add_subplot(111, aspect='equal', autoscale_on=False,
                                            xlim=(-0.2, limits), ylim=(-0.2, limits))

        self.rect = plt.Rectangle((0, 0),
                                  self.simulationConstants["environmentSideLength"],
                                  self.simulationConstants["environmentSideLength"],
                                  ec='none', lw=2, fc='none')
        self.axis.add_patch(self.rect)
        plt.grid()

        # initialize plot with empty position data, cyclic colormap and swimmer phase as data for the color map
        initialState = self.states[0]
        initialX = initialState[:, 0]
        initialY = initialState[:, 1]
        initialPhi = initialState[:, 2]
        self.swimmerPlot = self.axis.quiver(initialX, initialY, np.cos(initialPhi), np.sin(initialPhi),
                                            initialPhi % 2 * np.pi, cmap=plt.get_cmap("twilight"), clim=[0, 2 * np.pi],
                                            pivot="middle", scale=250, units="width", width=0.0005, headwidth=30,
                                            headlength=50, headaxislength=45, minshaft=0.99, minlength=0)

        self.trajectoryLines = []
        if self.numSwimmers <= 10:
            for i in range(self.numSwimmers):
                trajectoryLine, = self.axis.plot([], [], "go", ms=0.75)
                self.trajectoryLines.append(trajectoryLine)

        self.figure.colorbar(self.swimmerPlot, ax=self.axis)

        def plotInit():
            # microswimmersPlot.set_offsets([], [])
            self.rect.set_edgecolor('none')
            for i in range(self.numSwimmers):
                if self.numSwimmers <= 10:
                    self.trajectoryLines[i].set_data([], [])

            # an array of all updates objects must be returned in this function, here the scatter plot, rect and all of the trajectory line plots
            iterableArtists = [self.swimmerPlot, self.rect] + self.trajectoryLines
            return iterableArtists

        def animate(t):
            data = self.states[t]

            markerSize = int(self.figure.dpi * 2 * self.simulationConstants["swimmerSize"] * self.figure.get_figwidth()
                             / np.diff(self.axis.get_xbound())[0])
            areaSize = np.ones(self.numSwimmers) * 0.5 * markerSize ** 2

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
            self.swimmerPlot.set_UVC(np.cos(phi), np.sin(phi), phi % 2 * np.pi)

            for i in range(self.numSwimmers):
                if self.numSwimmers <= 10:
                    self.trajectoryLines[i].set_data(self.states[:t, i, 0], self.states[:t, i, 1])

            # ax.text(0.5, 0.5, "Zeit: {} s".format(i / CONSTANTS["frames"]))

            # an array of all updates objects must be returned in this function, here the scatter plot, rect and all of the trajectory line plots
            iterableArtists = [self.swimmerPlot, self.rect] + self.trajectoryLines
            return iterableArtists

        ani = animation.FuncAnimation(self.figure, func=animate, frames=self.simulationConstants["timeSteps"],
                                      interval=1000 / 60, blit=True, init_func=plotInit)

        if self.simulationConstants['saveVideo']:
            mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\konst\\Desktop\\ffmpeg\\bin\\ffmpeg.exe'
            f = r"c:\\Users\\konst\\Desktop\\animation.mp4"
            writervideo = animation.FFMpegWriter(fps=60, bitrate=3000, codec='h264_nvenc')
            ani.save(
                f, writer=writervideo,
                progress_callback=lambda i, n: printProgressBar(i, n, prefix='Animation Progress:',
                                                                suffix='Animation Complete', length=50)
            )

        plt.show()
