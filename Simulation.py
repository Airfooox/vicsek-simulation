import random

import uuid
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import curve_fit
from numba import uint16, uint32, float32, types, njit
from numba.experimental import jitclass
mpl.use("TkAgg")

import util
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

    def addSwimmer(self, cellIndex: int, swimmerId: int):
        self.cells[cellIndex] = np.append(self.cells[cellIndex], np.uint16(swimmerId))
        # self.cells[cellIndex].append(np.uint16(swimmerId))

    def removeSwimmer(self, cellIndex: int, swimmerId: int):
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

    def clear(self):
        self.cells = [np.zeros(0, dtype=np.uint16) for _ in range(self.rows * self.columns)]


class Simulation:
    def __init__(self, simulationIndex, numSimulation, simulationConfig, timePercentageUsedForMean):
        self.simulationConfig = simulationConfig
        self.framesUsedForMean = int(np.ceil(
            ((1 - (timePercentageUsedForMean / 100)) * self.simulationConfig["timeSteps"])))

        self.grid = Grid(self.simulationConfig['environmentSideLength'], self.simulationConfig['environmentSideLength'], self.simulationConfig['interactionRadius'])

        # initialize the swimmers with position, angle of velocity, amplitude, period and phase shift
        initialSwimmerState = []
        for groupIdentifier, groupConfig in self.simulationConfig['groups'].items():
            for swimmerIndex in range(groupConfig["numSwimmers"]):
                xPos = np.random.rand() * self.simulationConfig["environmentSideLength"]
                yPos = np.random.rand() * self.simulationConfig["environmentSideLength"]
                vPhi = np.random.rand() * 2 * np.pi  # angle in rad
                oscillationAmplitude = groupConfig["oscillationAmplitude"]
                oscillationPeriod = groupConfig["oscillationPeriod"]
                oscillationPhaseShift = groupConfig["oscillationPhaseShift"]
                initialSwimmerState.append(
                    [xPos, yPos, vPhi, oscillationAmplitude, oscillationPeriod, oscillationPhaseShift])

        self.numSwimmers = len(initialSwimmerState)
        # initialize states array and add initial values for first frame
        self.states = np.zeros((self.simulationConfig["timeSteps"], self.numSwimmers, 6), dtype=np.float64)
        self.states[0] = np.array(initialSwimmerState, dtype=np.float64)
        self.initializeGrid()

        self.absoluteVelocities = np.zeros((self.simulationConfig["timeSteps"]), dtype=np.float64)
        self.absoluteGroupVelocities = np.zeros((self.simulationConfig["timeSteps"], len(self.simulationConfig['groups'].values())), dtype=np.float64)
        self.vectorialVelocities = np.zeros((self.simulationConfig["timeSteps"], 2), dtype=np.float64)
        self.vectorialGroupVelocities = np.zeros((self.simulationConfig["timeSteps"], len(self.simulationConfig['groups'].values()), 2), dtype=np.float64)
        self.absoluteGroupVelocityConfig = np.array(list(map(lambda x: x['numSwimmers'], self.simulationConfig['groups'].values())), dtype=np.int16)

        self.totalAbsoluteVelocity = 0
        self.totalAbsoluteGroupVelocities = np.zeros(len(self.absoluteGroupVelocityConfig), dtype=np.float64)
        self.totalVectorialVelocity = np.zeros(2, dtype=np.float64)
        self.totalVectorialGroupVelocities = np.zeros((len(self.absoluteGroupVelocityConfig), 2), dtype=np.float64)

    def initializeGrid(self):
        self.grid.clear()
        for swimmerIndex, initialSwimmerData in enumerate(self.states[0]):
            xPos, yPos = initialSwimmerData[0], initialSwimmerData[1]
            self.grid.addSwimmer(self.grid.getCellIndex(xPos, yPos), swimmerIndex)

    def simulate(self):
        for t in np.arange(1, self.simulationConfig["timeSteps"]):
            previousState = self.states[t - 1]

            args = [self.numSwimmers, self.grid, previousState, self.absoluteGroupVelocityConfig,
                    self.simulationConfig['environmentSideLength'], self.simulationConfig['randomAngleAmplitude'],
                    self.simulationConfig['interactionRadius'], self.simulationConfig['interactionStrengthFactor'], self.simulationConfig['velocity']]
            self.grid, self.states[t], self.vectorialVelocities[t], self.vectorialGroupVelocities[t], self.absoluteVelocities[t], self.absoluteGroupVelocities[t] = self.calculateNewState(t, *args)

        self.totalAbsoluteVelocity = Simulation.calculateAbsoluteVelocityTotal(
            self.simulationConfig["timeSteps"], self.framesUsedForMean,
            self.absoluteVelocities)
        self.totalVectorialVelocity = np.mean(self.vectorialVelocities[self.framesUsedForMean:], axis=0)

        for groupIndex in np.arange(len(self.absoluteGroupVelocityConfig)):
            self.totalAbsoluteGroupVelocities[groupIndex] = Simulation.calculateAbsoluteVelocityTotal(
                self.simulationConfig["timeSteps"], self.framesUsedForMean,
                self.absoluteGroupVelocities[:, groupIndex])

            self.totalVectorialGroupVelocities[groupIndex] = np.mean(self.vectorialGroupVelocities[self.framesUsedForMean:, groupIndex], axis=0)

        # printProgressBar(t, self.simulationConfig["timeSteps"] - 1, prefix='Simulation Progress:', suffix='Simulation Complete', length=50)

    @staticmethod
    @njit
    def calculateNewState(t, numSwimmers, grid, previousState, absoluteGroupVelocityConfig, environmentSideLength, randomAngleAmplitude, interactionRadius, interactionStrengthFactor, velocity):
        vectorialSumVelocity = np.array([0, 0], dtype=np.float64)

        absoluteGroupVelocityIndex = 0
        absoluteVelocityGroupCount = absoluteGroupVelocityConfig[absoluteGroupVelocityIndex]
        vectorialGroupSumVelocities = np.zeros((len(absoluteGroupVelocityConfig), 2), dtype=np.float64)

        newState = previousState.copy()
        for swimmerIndex in np.arange(numSwimmers):
            swimmerState = newState[swimmerIndex]
            interactionSwimmerIndices = np.zeros(0, dtype=np.uint16)

            # potential interaction swimmer indices
            gridInteractionSwimmerIndices = grid.getNeighbourCellsSwimmerIndices(swimmerState[0], swimmerState[1])
            for interactionSwimmerIndex in gridInteractionSwimmerIndices:
                # no self interaction, because we already take the current state as the base values
                if interactionSwimmerIndex == swimmerIndex:
                    continue

                interactionSwimmerState = previousState[interactionSwimmerIndex]
                distanceBetween = (swimmerState[0] - interactionSwimmerState[0]) ** 2 + (
                            swimmerState[1] - interactionSwimmerState[1]) ** 2
                if distanceBetween <= interactionRadius ** 2:
                    interactionSwimmerIndices = np.append(interactionSwimmerIndices, interactionSwimmerIndex)
                    # print("INTERACTION:", t, swimmerIndex, interactionSwimmerIndex)

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
                gridShadowInteractionSwimmerIndices = grid.getNeighbourCellsSwimmerIndices(shadowSwimmerPosition[0],
                                                                                           shadowSwimmerPosition[1])
                for shadowInteractionSwimmerIndex in gridShadowInteractionSwimmerIndices:
                    # not interacting with themselves
                    if shadowInteractionSwimmerIndex == swimmerIndex:
                        continue

                    # dont duplicate
                    # print("SHADOW", t, swimmerIndex, shadowInteractionSwimmerIndex, shadowInteractionSwimmerIndex in interactionSwimmerIndices, interactionSwimmerIndices)
                    if shadowInteractionSwimmerIndex in interactionSwimmerIndices:
                        continue


                    shadowInteractionSwimmerState = previousState[shadowInteractionSwimmerIndex]
                    distanceBetween = (shadowSwimmerPosition[0] - shadowInteractionSwimmerState[0]) ** 2 + (
                            shadowSwimmerPosition[1] - shadowInteractionSwimmerState[1]) ** 2
                    if distanceBetween ** 2 <= interactionRadius:
                        interactionSwimmerIndices = np.append(interactionSwimmerIndices,
                                                              shadowInteractionSwimmerIndex)
                        # print("SHADOW INTERACTION:", t, swimmerIndex, shadowInteractionSwimmerIndex, distanceBetween)
            # print(interactionSwimmerIndices)

            # STANDARD VICSEK INTERACTION
            # i and j are in range
            # sinSum = np.sin(swimmerState[2])
            # cosSum = np.cos(swimmerState[2])
            # for interactionSwimmerIndex in interactionSwimmerIndices:
            #     interactionSwimmerState = previousState[interactionSwimmerIndex]
            #     sinSum += np.sin(interactionSwimmerState[2])
            #     cosSum += np.cos(interactionSwimmerState[2])
            #
            # averageAngle = np.arctan2(sinSum, cosSum)
            # randomAngle = ((np.random.rand() - 0.5) * randomAngleAmplitude)
            # cosinesOscillation = swimmerState[3] * np.cos(
            #     (2 * np.pi / swimmerState[4]) * t + swimmerState[5])
            # # print(averageAngle, cosinesOscillation, averageAngle + randomAngle + cosinesOscillation)
            # swimmerState[2] = averageAngle + randomAngle + cosinesOscillation

            # ORIENTATION POTENTIAL
            preFactor = -interactionStrengthFactor / np.pi
            sum = 0
            for interactionSwimmerIndex in interactionSwimmerIndices:
                interactionSwimmerState = previousState[interactionSwimmerIndex]
                sum += -np.sin(swimmerState[2] - interactionSwimmerState[2])

            randomAngle = ((np.random.rand() - 0.5) * randomAngleAmplitude)
            cosinesOscillation = swimmerState[3] * np.cos(
                (2 * np.pi / swimmerState[4]) * t + swimmerState[5])
            # print(averageAngle, cosinesOscillation, averageAngle + randomAngle + cosinesOscillation)
            swimmerState[2] += -preFactor * sum + randomAngle + cosinesOscillation

            xVelCos = np.cos(swimmerState[2])
            yVelSin = np.sin(swimmerState[2])
            velVec = np.array([xVelCos, yVelSin], dtype=np.float64)
            vectorialSumVelocity += velVec

            # sum velocity vectors for every group
            vectorialGroupSumVelocities[absoluteGroupVelocityIndex] += velVec
            absoluteVelocityGroupCount -= 1
            if absoluteVelocityGroupCount <= 0:
                absoluteGroupVelocityIndex = min(absoluteGroupVelocityIndex + 1,
                                                 len(absoluteGroupVelocityConfig) - 1)
                absoluteVelocityGroupCount = absoluteGroupVelocityConfig[absoluteGroupVelocityIndex]

            swimmerState[0] += velocity * xVelCos
            swimmerState[1] += velocity * yVelSin

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

            previousSwimmerGridIndex = grid.getCellIndex(previousState[swimmerIndex][0],
                                                         previousState[swimmerIndex][1])
            newSwimmerGridIndex = grid.getCellIndex(swimmerState[0], swimmerState[1])
            if previousSwimmerGridIndex != newSwimmerGridIndex:
                grid.removeSwimmer(previousSwimmerGridIndex, swimmerIndex)
                grid.addSwimmer(newSwimmerGridIndex, swimmerIndex)

        absoluteVelocity = np.linalg.norm(vectorialSumVelocity) / numSwimmers
        vectorialVelocity = vectorialSumVelocity / np.linalg.norm(vectorialSumVelocity)

        absoluteGroupVelocity = np.zeros(len(absoluteGroupVelocityConfig), dtype=np.float64)
        vectorialGroupVelocity = np.zeros((len(absoluteGroupVelocityConfig), 2), dtype=np.float64)
        for index, numGroupSwimmers in enumerate(absoluteGroupVelocityConfig):
            absoluteGroupVelocity[index] = np.linalg.norm(vectorialGroupSumVelocities[index]) / numGroupSwimmers
            vectorialGroupVelocity[index] = vectorialGroupSumVelocities[index] / np.linalg.norm(vectorialGroupSumVelocities[index])


        return grid, newState, vectorialVelocity, vectorialGroupVelocity, absoluteVelocity, absoluteGroupVelocity

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
            absoluteVelocity = np.mean(absoluteVelocities[framesUsedForMean:])
        except ZeroDivisionError:
            absoluteVelocity = -2
            # print(timeSteps, np.array(absoluteVelocities))
        return absoluteVelocity

    def animate(self, showGroup = False, saveVideo = False, videoPath = None, fixedTimeStep = None, saveFixedTimeSetPictureDir = None):
        if fixedTimeStep and fixedTimeStep >= self.simulationConfig['timeSteps']:
            print('Given fixed timestep is not in the bounds of the simulation')
            return

        # mpl.rcParams['text.usetex'] = True
        # mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']  # for \text command

        # initialize animation
        self.figure = plt.figure()
        self.figure.subplots_adjust(left=0, right=0.95, bottom=0.05, top=0.95)

        padding = 0
        textPadding = (0.05 + 0.06 * len(self.absoluteGroupVelocityConfig)) * self.simulationConfig['environmentSideLength']
        self.axis = self.figure.add_subplot(111, aspect='equal', autoscale_on=False,
                                            xlim=(-padding, self.simulationConfig['environmentSideLength'] + padding),
                                            ylim=(-padding, self.simulationConfig['environmentSideLength'] + padding + textPadding))

        self.rect = plt.Rectangle((0, 0),
                                  self.simulationConfig["environmentSideLength"],
                                  self.simulationConfig["environmentSideLength"],
                                  ec='none', lw=2, fc='none')
        self.rect.set_edgecolor('k')
        self.axis.add_patch(self.rect)
        # plt.grid()


        if not fixedTimeStep:
            # initialize plot with empty position data, cyclic colormap and swimmer phase as data for the color map
            initialState = self.states[0]
        else:
            initialState = self.states[fixedTimeStep]

        initialX = initialState[:, 0]
        initialY = initialState[:, 1]
        initialPhi = initialState[:, 2]

        norm = None
        colormap = None
        if not showGroup:
            colorData = initialPhi % (2 * np.pi)
            clim = [0, 2 * np.pi]
            norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
            colormap = plt.get_cmap("twilight")
        else:
            colorData = np.zeros(0, dtype=np.uint16)
            # clim = [0, len(self.absoluteGroupVelocityConfig) + 1]
            clim = None
            boundaries = []
            for index, groupNumSwimmers in enumerate(self.absoluteGroupVelocityConfig):
                boundaries.append(index + 1)
                colorData = np.append(colorData, np.zeros(groupNumSwimmers, dtype=np.uint16) + index + 1)
            norm = mpl.colors.Normalize(vmin=boundaries[0], vmax=boundaries[-1])
            colormap = plt.get_cmap("winter")



        self.swimmerPlot = self.axis.quiver(initialX, initialY, np.cos(initialPhi), np.sin(initialPhi),
                                            colorData, cmap=colormap, norm=norm, clim=clim,
                                            pivot="middle", scale=250, units="width", width=0.0001, headwidth=20,
                                            headlength=50, headaxislength=45, minshaft=0.99, minlength=0)
        self.swimmerPlot.set_sizes(np.ones(self.numSwimmers) * 50)

        self.trajectoryLines = []
        if self.numSwimmers <= 10:
            for i in range(self.numSwimmers):
                trajectoryLine, = self.axis.plot([], [], "go", ms=0.75)
                self.trajectoryLines.append(trajectoryLine)

        if not showGroup:
            colorbar = self.figure.colorbar(self.swimmerPlot, ax=self.axis)
            colorbar.set_label(r'orientation')

            majorTicks = util.Multiple(2)
            colorbar.ax.yaxis.set_major_locator(majorTicks.locator())
            colorbar.ax.yaxis.set_major_formatter(majorTicks.formatter())

        rho = np.round(self.numSwimmers / self.simulationConfig['environmentSideLength']**2, 2)
        generalSimulationConfigString = r'$N_{total}=%s, \varrho=%s, \eta=%s, g=%s, r=%s$' %\
                                        (self.numSwimmers, rho, self.simulationConfig['randomAngleAmplitude'],
                                         self.simulationConfig['interactionStrengthFactor'], self.simulationConfig['interactionRadius'])
        self.axis.text(
            0,
            self.simulationConfig["environmentSideLength"] + textPadding
            - 0.04 * self.simulationConfig["environmentSideLength"],
            generalSimulationConfigString)

        for index, groupConfig in self.simulationConfig['groups'].items():
            groupNumSwimmers = groupConfig['numSwimmers']
            amplitude = groupConfig['oscillationAmplitude']
            period = groupConfig['oscillationPeriod']
            phaseShift = groupConfig['oscillationPhaseShift']
            amplitudeString = util.multipleFormatter(2 ** 16, np.pi)(amplitude, None)
            phaseShiftString = util.multipleFormatter(2 ** 16, np.pi)(phaseShift, None)
            groupsSimulationConfigString = r'Gruppe $%s: (N=%s, A=$%s$, T=%s, \Delta \varphi=$%s$)$' % \
                                            (index, groupNumSwimmers, amplitudeString, period, phaseShiftString) + '\n'

            color = None
            if showGroup:
                color = colormap(norm(int(index)))

            self.axis.text(
                0,
                self.simulationConfig["environmentSideLength"] + textPadding
                - (0.05 * (int(index) + 2)) * self.simulationConfig["environmentSideLength"],
                groupsSimulationConfigString, color=color)

        if fixedTimeStep:
            for trajectoryLine in self.trajectoryLines:
                trajectoryLine.set_data(self.states[:fixedTimeStep, i, 0], self.states[:fixedTimeStep, i, 1])

            if saveFixedTimeSetPictureDir:
                plt.savefig(saveFixedTimeSetPictureDir, bbox_inches='tight')
                plt.close(self.figure)
            else:
                plt.show()
            return

        def plotInit():
            for i in range(self.numSwimmers):
                if self.numSwimmers <= 10:
                    self.trajectoryLines[i].set_data([], [])

            # an array of all updates objects must be returned in this function, here the scatter plot, rect and all of the trajectory line plots
            iterableArtists = [self.swimmerPlot, self.rect] + self.trajectoryLines
            return iterableArtists

        def animate(t):
            data = self.states[t]

            # update pieces of the animation
            x = data[:, 0]
            y = data[:, 1]
            phi = data[:, 2]
            # combine x and y data of objects in following way:
            # [x1, x2, ..., xn] and [y1, y2, ..., yn] -> [[x1, y1], [x2,y2], ..., [xn, yn]]
            self.swimmerPlot.set_offsets(np.c_[x, y])

            if not showGroup:
                updatedColorData = phi % (2 * np.pi)
            else:
                updatedColorData = np.zeros(0, dtype=np.uint16)
                for index, groupNumSwimmers in enumerate(self.absoluteGroupVelocityConfig):
                    updatedColorData = np.append(updatedColorData, np.zeros(groupNumSwimmers, dtype=np.uint16) + index + 1)
            self.swimmerPlot.set_UVC(np.cos(phi), np.sin(phi), updatedColorData)

            for i in range(self.numSwimmers):
                if self.numSwimmers <= 10:
                    self.trajectoryLines[i].set_data(self.states[:t, i, 0], self.states[:t, i, 1])

            # ax.text(0.5, 0.5, "Zeit: {} s".format(i / CONSTANTS["frames"]))

            # an array of all updates objects must be returned in this function, here the scatter plot, rect and all of the trajectory line plots
            iterableArtists = [self.swimmerPlot, self.rect] + self.trajectoryLines
            return iterableArtists

        ani = animation.FuncAnimation(self.figure, func=animate, frames=self.simulationConfig["timeSteps"],
                                      interval=1000 / 60, blit=True, init_func=plotInit)

        if self.simulationConfig['saveVideo'] or saveVideo:
            if not videoPath:
                videoPath = rf'C:\Users\konst\OneDrive\Uni\Anstellung\Prof. Menzel (2020-22)\vicsek\simulation\videos\{uuid.uuid4()}'
            mpl.rcParams['animation.ffmpeg_path'] = r'C:\Users\konst\Desktop\ffmpeg\bin\ffmpeg.exe'
            writervideo = animation.FFMpegWriter(fps=30, bitrate=5000)
            ani.save(
                videoPath, writer=writervideo,
                progress_callback=lambda i, n: printProgressBar(i, n, prefix='Animation Progress:',
                                                                suffix='Animation Complete', length=50)
            )
        else:
            plt.show()
