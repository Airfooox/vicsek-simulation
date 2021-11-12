import numpy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import pdist, squareform
from numba import njit

import util
from util import printProgressBar
import time


class Simulation:
    def __init__(self, CONSTANTS):
        self.simulationConstants = CONSTANTS
        self.numFrames = self.simulationConstants["fps"] * self.simulationConstants["time"]
        self.tau = 1 / self.simulationConstants["fps"]

        # initialize the swimmers with position and angle of velocity
        self.swimmers = []
        # self.swimmers.append([0.9, 0.9, 180 * np.pi / 180])
        # self.swimmers.append([-0.9, -0.9, 0 * np.pi / 180])
        for i in range(self.simulationConstants["numSwimmers"]):
            xPos = np.random.rand() * self.simulationConstants["environmentSideLength"] - self.simulationConstants["environmentSideLength"] / 2
            yPos = np.random.rand() * self.simulationConstants["environmentSideLength"] - self.simulationConstants["environmentSideLength"] / 2
            vPhi = np.random.rand() * 2 * np.pi # angle in rad
            self.swimmers.append([xPos, yPos, vPhi])

        self.swimmerColors = np.random.rand(self.simulationConstants["numSwimmers"])

        # initialize states array and add initial values for first frame
        self.states = np.empty((self.numFrames, self.simulationConstants["numSwimmers"], 3), dtype=numpy.float64)
        for i in range(self.simulationConstants["numSwimmers"]):
            swimmerState = np.array([[self.swimmers[i][0], self.swimmers[i][1], self.swimmers[i][2]]], dtype=numpy.float64)
            self.states[0][i] = swimmerState

    def simulate(self):
        for t in range (1, self.numFrames):
            # time1 = time.time()
            self.states[t] = self.simulationStep(t)
            # time2 = time.time()
            # print('{:d} timestep took {:.3f} ms'.format(t, (time2 - time1) * 1000.0))

            # printProgressBar(t, self.numFrames - 1, prefix='Simulation Progress:', suffix='Simulation Complete', length=50)

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
        oscillationAmplitude = self.simulationConstants["oscillationAmplitude"]
        fps = self.simulationConstants["fps"]
        initialVelocity = self.simulationConstants["initialVelocity"]
        return self.calcNewState(t, previousState, newState, index1, index2, numSwimmers, environmentSideLength, interactionRadius, randomAngleAmplitude, oscillationAmplitude, fps, initialVelocity)

    @staticmethod
    @njit
    def calcNewState(t, previousState, newState, index1, index2, numSwimmers, environmentSideLength, interactionRadius, randomAngleAmplitude, oscillationAmplitude, fps, initialVelocity):
        tau = 1 / fps
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

            shadowSwimmerPositions = np.empty((0), dtype=numpy.float64)
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

            averageAngle = np.arctan2(sinSum, cosSum)
            randomAngle = (np.random.rand() - 0.5) * randomAngleAmplitude
            sinusOscillation = oscillationAmplitude * np.sin(
                (t / (2 * fps)) * 2 * np.pi)
            swimmerState[2] = averageAngle + randomAngle + sinusOscillation

            xVel = initialVelocity * np.cos(swimmerState[2])
            yVel = initialVelocity * np.sin(swimmerState[2])

            swimmerState[0] += xVel * tau
            swimmerState[1] += yVel * tau

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

        return newState


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

        self.swimmerPlot = self.axis.scatter([], [])

        def plotInit():
            # microswimmersPlot.set_offsets([], [])
            self.rect.set_edgecolor('none')
            return self.swimmerPlot, self.rect

        def animate(i):
            data = self.states[i]

            markerSize = int(self.figure.dpi * 2 * self.simulationConstants["swimmerSize"] * self.figure.get_figwidth()
                             / np.diff(self.axis.get_xbound())[0])
            areaSize = np.ones(self.simulationConstants["numSwimmers"]) * 0.5 * markerSize ** 2

            # update pieces of the animation
            self.rect.set_edgecolor('k')

            x = data[:, 0]
            y = data[:, 1]
            # combine x and y data of objects in following way:
            # [x1, x2, ..., xn] and [y1, y2, ..., yn] -> [[x1, y1], [x2,y2], ..., [xn, yn]]
            self.swimmerPlot.set_offsets(np.c_[x, y])
            self.swimmerPlot.set_sizes(areaSize)
            self.swimmerPlot.set_array(self.swimmerColors)

            # ax.text(0.5, 0.5, "Zeit: {} s".format(i / CONSTANTS["frames"]))
            return self.swimmerPlot, self.rect

        ani = animation.FuncAnimation(self.figure, animate, frames=self.numFrames,
                                      interval=1000 / self.simulationConstants["fps"], blit=True, init_func=plotInit)

        # mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\konst\\Desktop\\ffmpeg\\bin\\ffmpeg.exe'
        # f = r"c:\\Users\\konst\\Desktop\\animation.mp4"
        # writervideo = animation.FFMpegWriter(fps=self.simulationConstants["fps"], bitrate=3000)
        # ani.save(
        #     f, writer=writervideo,
        #     progress_callback=lambda i, n: printProgressBar(i, n, prefix='Animation Progress:',
        #                                                     suffix='Animation Complete', length=50)
        # )

        plt.show()

