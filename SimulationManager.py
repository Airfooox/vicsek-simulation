import os
import json
import numpy as np
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool

from Simulation import Simulation
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

        pool = ProcessingPool(nodes=(mp.cpu_count() - 2))
        for _ in tqdm(pool.imap(SimulationManager.runSimulation, simulationScenarios), total=len(simulationScenarios)):
            pass
        pool.close()
        pool.join()

    @staticmethod
    def runSimulation(simulationScenario):
        simulationIndex = simulationScenario.simulationIndex
        numSimulation = simulationScenario.numSimulation
        scenarioDataDir = simulationScenario.scenarioDataDir
        scenarioConfig = simulationScenario.scenarioConfig
        saveTrajectoryData = simulationScenario.saveTrajectoryData

        simulation = Simulation(simulationIndex, numSimulation, scenarioConfig)
        simulation.simulate()
        absoluteVelocities = simulation.absoluteVelocities
        absoluteVelocity = simulation.calculateAbsoluteVelocity()

        if not (os.path.exists(scenarioDataDir) and os.path.isdir(scenarioDataDir)):
            os.mkdir(scenarioDataDir)

        with open(os.path.join(scenarioDataDir, 'config.txt'), 'w') as configFile:
            json.dump(scenarioConfig, configFile)

        with open(os.path.join(scenarioDataDir, 'absoluteVelocities.npy'), 'wb') as absoluteVelocitiesFile:
            np.save(absoluteVelocitiesFile, absoluteVelocities)

        with open(os.path.join(scenarioDataDir, 'absoluteVelocity.txt'), 'w') as absoluteVelocityFile:
            json.dump(absoluteVelocity, absoluteVelocityFile)

        if saveTrajectoryData or absoluteVelocity <= 0:
            statesData = simulation.states
            print(absoluteVelocity, os.path.join(scenarioDataDir, 'statesData.npy'))
            with open(os.path.join(scenarioDataDir, 'statesData.npy'), 'wb') as statesFile:
                np.save(statesFile, statesData)


class SimulationGroup:
    def __init__(self, simulationDataDir, configFunction, numSimulation, repeatNum, timePercentageUsedForMean, saveTrajectoryData = False):
        self.simulationDataDir = simulationDataDir
        self.constantsFunc = configFunction
        self.numSimulation = numSimulation
        self.repeatNum = repeatNum
        self.timePercentageUsedForMean = timePercentageUsedForMean
        self.saveTrajectoryData = saveTrajectoryData

        if not (os.path.exists(self.simulationDataDir) and os.path.isdir(self.simulationDataDir)):
            os.mkdir(self.simulationDataDir)

        simulationGroupConfig = {
            'numSimulation': self.numSimulation,
            'repeatNum': self.repeatNum,
            'timePercentageUsedForMean': self.timePercentageUsedForMean,
            'saveTrajectoryData': self.saveTrajectoryData,
        }
        with open(os.path.join(simulationDataDir, "simulationGroupConfig.txt"), "w") as simulationGroupConfigFile:
            json.dump(simulationGroupConfig, simulationGroupConfigFile)

        self.simulationScenarios = []
        for simulationIndex in range(numSimulation):
            for subSimulationIndex in range(repeatNum):
                scenarioDataDir = self.simulationDataDir + '/' + str(simulationIndex) + '_' + str(subSimulationIndex)
                self.simulationScenarios.append(SimulationScenario(simulationIndex, numSimulation, scenarioDataDir, self.constantsFunc(simulationIndex, numSimulation), self.timePercentageUsedForMean, self.saveTrajectoryData))


class SimulationScenario:
    def __init__(self, simulationIndex, numSimulation, scenarioDataDir, scenarioConfig, timePercentageUsedForMean, saveTrajectoryData):
        self.simulationIndex = simulationIndex
        self.numSimulation = numSimulation
        self.scenarioDataDir = scenarioDataDir
        self.scenarioConstants = scenarioConfig
        self.timePercentageUsedForMean = timePercentageUsedForMean
        self.saveTrajectoryData = saveTrajectoryData