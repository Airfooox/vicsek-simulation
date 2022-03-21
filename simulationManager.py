import os
import json
import numpy as np
import multiprocessing as mp

from classes import Simulation
from util import printProgressBar
from tqdm import tqdm

class SimulationManager:
    def __init__(self, dataDir):
        self.dataDir = dataDir
        self.simulationGroups = []

        if not (os.path.exists(self.dataDir) and os.path.isdir(self.dataDir)):
            os.mkdir(self.dataDir)

    def appendGroup(self, simulationGroup):
        self.simulationGroups.append(simulationGroup)

    def simulate(self):
        simulationScenarios = []
        for simulationGroup in self.simulationGroups:
            simulationScenarios = simulationScenarios + simulationGroup.simulationScenarios

        pool = mp.Pool(processes=(mp.cpu_count() - 2))
        for _ in tqdm(pool.imap(func=SimulationManager.runSimulation, iterable=simulationScenarios), total=len(simulationScenarios)):
            pass
        pool.close()
        pool.join()

    @staticmethod
    def runSimulation(simulationScenario):
        scenarioDataDir = simulationScenario.scenarioDataDir
        scenarioConstants = simulationScenario.scenarioConstants
        saveTrajectoryData = simulationScenario.saveTrajectoryData

        simulation = Simulation(scenarioConstants)
        simulation.simulate()
        absoluteVelocities = simulation.getAbsoluteVelocities()
        absoluteVelocity = simulation.getAbsoluteVelocityTotal()

        if not (os.path.exists(scenarioDataDir) and os.path.isdir(scenarioDataDir)):
            os.mkdir(scenarioDataDir)

        with open(scenarioDataDir + '/constants.txt', 'w') as constantsFile:
            json.dump(scenarioConstants, constantsFile)

        with open(scenarioDataDir + '/absoluteVelocities.npy', 'wb') as absoluteVelocitiesFile:
            np.save(absoluteVelocitiesFile, absoluteVelocities)

        with open(scenarioDataDir + '/absoluteVelocity.txt', 'w') as absoluteVelocityFile:
            json.dump(absoluteVelocity, absoluteVelocityFile)

        if saveTrajectoryData or absoluteVelocity <= 0:
            statesData = simulation.getStates()
            print(absoluteVelocity, scenarioDataDir + '/statesData.npy')
            with open(scenarioDataDir + '/statesData.npy', 'wb') as statesFile:
                np.save(statesFile, statesData)


class SimulationGroup:
    def __init__(self, simulationDataDir, constantsFunc, numSimulation, repeatNum, saveTrajectoryData = False, timeSteps = 3600):
        self.simulationDataDir = simulationDataDir
        self.constantsFunc = constantsFunc
        self.numSimulation = numSimulation
        self.repeatNum = repeatNum
        self.saveTrajectoryData = saveTrajectoryData
        self.timeSteps = timeSteps

        if not (os.path.exists(self.simulationDataDir) and os.path.isdir(self.simulationDataDir)):
            os.mkdir(self.simulationDataDir)

        self.simulationScenarios = []
        for i in range(numSimulation):
            for j in range(repeatNum):
                scenarioDataDir = self.simulationDataDir + '/' + str(i) + '_' + str(j)
                self.simulationScenarios.append(SimulationScenario(scenarioDataDir, self.constantsFunc(i, numSimulation, timeSteps), self.saveTrajectoryData))


class SimulationScenario:
    def __init__(self, scenarioDataDir, scenarioConstants, saveTrajectoryData):
        self.scenarioDataDir = scenarioDataDir
        self.scenarioConstants = scenarioConstants
        self.saveTrajectoryData = saveTrajectoryData