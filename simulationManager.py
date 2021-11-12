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

        simulation = Simulation(scenarioConstants)
        simulation.simulate()
        statesData = simulation.states

        if not (os.path.exists(scenarioDataDir) and os.path.isdir(scenarioDataDir)):
            os.mkdir(scenarioDataDir)

        with open(scenarioDataDir + '/constants.txt', 'w') as constantsFile:
            json.dump(scenarioConstants, constantsFile)
        np.save(scenarioDataDir + '/statesData', statesData)


class SimulationGroup:
    def __init__(self, simulationDataDir, constantsFunc, numSimulation, repeatNum, time=60, fps=30):
        self.simulationDataDir = simulationDataDir
        self.constantsFunc = constantsFunc
        self.numSimulation = numSimulation
        self.repeatNum = repeatNum
        self.time = time
        self.fps = fps

        if not (os.path.exists(self.simulationDataDir) and os.path.isdir(self.simulationDataDir)):
            os.mkdir(self.simulationDataDir)

        self.simulationScenarios = []
        for i in range(numSimulation):
            for j in range(repeatNum):
                scenarioDataDir = self.simulationDataDir + '/' + str(i) + '_' + str(j)
                self.simulationScenarios.append(SimulationScenario(scenarioDataDir, self.constantsFunc(i, numSimulation, time, fps)))


class SimulationScenario:
    def __init__(self, scenarioDataDir, scenarioConstants):
        self.scenarioDataDir = scenarioDataDir
        self.scenarioConstants = scenarioConstants