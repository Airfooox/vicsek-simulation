import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from util import printProgressBar


class Simulation:
    def __init__(self, CONSTANTS):
        self.simulationConstants = CONSTANTS
        self.numFrames = self.simulationConstants["fps"] * self.simulationConstants["time"]
        self.tau = 1 / self.simulationConstants["fps"]

        # initialize the swimmers with position and angle of velocity
        self.swimmers = []
        for i in range(self.simulationConstants["numSwimmers"]):
            xPos = np.random.rand() * self.simulationConstants["environmentSideLength"] - self.simulationConstants["environmentSideLength"] / 2
            yPos = np.random.rand() * self.simulationConstants["environmentSideLength"] - self.simulationConstants["environmentSideLength"] / 2
            vPhi = np.random.rand() * 2 * np.pi
            self.swimmers.append([xPos, yPos, vPhi])

        self.swimmerColors = np.random.rand(self.simulationConstants["numSwimmers"])

        # initialize states array and add initial values for first frame
        self.states = np.empty((self.numFrames, self.simulationConstants["numSwimmers"], 3), dtype=float)
        for i in range(self.simulationConstants["numSwimmers"]):
            swimmerState = np.array([[self.swimmers[i][0], self.swimmers[i][1], self.swimmers[i][2]]], dtype=float)
            self.states[0][i] = swimmerState

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

    def simulate(self):
        for t in range (1, self.numFrames):
            self.states[t] = self.simulationStep(t)
            printProgressBar(t, self.numFrames - 1, prefix='Simulation Progress:', suffix='Simulation Complete', length=50)

        return self.states

    def simulationStep(self, t):
        previousState = []
        try:
            previousState = self.states[t - 1]
        except IndexError:
            previousState = self.simulationStep(t - 1)

        newState = previousState.copy()

        for i in range(self.simulationConstants["numSwimmers"]):
            swimmerState = newState[i]

            xVel = self.simulationConstants["initialVelocity"] * np.cos(swimmerState[2])
            yVel = self.simulationConstants["initialVelocity"] * np.sin(swimmerState[2])

            swimmerState[0] += xVel * self.tau
            swimmerState[1] += yVel * self.tau

        return newState

    def animate(self):
        def plotInit():
            # microswimmersPlot.set_offsets([], [])
            self.rect.set_edgecolor('none')
            return self.swimmerPlot, self.rect

        def animate(i):
            data = self.states[i]

            markerSize = int(self.figure.dpi * 2 * self.simulationConstants["swimmerSize"] * self.figure.get_figwidth()
                             / np.diff(self.axis.get_xbound())[0])
            areaSize = np.append(np.ones(self.simulationConstants["numSwimmers"]) * markerSize ** 2,
                                 np.ones(self.simulationConstants["numSwimmers"]) * 11 ** 2)

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

        plt.show()

