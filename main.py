# setup
import random

from Simulation import Simulation
from util import printProgressBar
import numpy as np
import time

from SimulationManager import SimulationManager, SimulationGroup

#TODO:
# Trajektorien von einzelnen Teilchen bei unterschiedlichen Parametern der Schwingung überprüfen
# Simulationsbox in Gitter mit Abstand des Interaktionsradius aufteilen, um Interaktionsberechnung effizienter zu machen


if __name__ == "__main__":
    starttime = time.perf_counter()

    # simulationDir = '/local/kzisiadis/vicsek-simulation'
    simulationDir = 'D:/simulationdata'
    simulationManager = SimulationManager(simulationDir)


    def initialSwimmerParametersameRhoConstants400PhaseShift90(simulationIndex, totalNumberOfSimulations, swimmerIndex):
        return {
            "oscillationAmplitude": np.pi / 40,
            "oscillationPeriod": 60,  # how many timesteps for one full oscillation
            "oscillationPhaseshift": random.choice([0, np.pi / 2])
        }
    def sameRhoConstants400PhaseShift90(i, numSimulation, defaultTimeSteps):
        return {
            "timeSteps": 7200,
            "timePercentageUsedForMean": 25,

            "environmentSideLength": 10,
            "numSwimmers": 400,
            "interactionRadius": 1,
            "randomAngleAmplitude": 15 * (i / numSimulation),

            "initialVelocity": 0.0025,
            "swimmerSize": 0.04,

            "saveVideo": False,
        }


    # sameRhoGroup400PhaseShift90 = SimulationGroup(simulationDataDir=simulationDir + '/sameRhoGroup400PhaseShift90',
    #                                   constantsFunc=sameRhoConstants400PhaseShift90, initialParameterFunc = initialSwimmerParametersameRhoConstants400PhaseShift90,
    #                                   numSimulation=100, repeatNum=100,
    #                                   saveTrajectoryData=False)
    # simulationManager.appendGroup(sameRhoGroup400PhaseShift90)

    def initialSwimmerParametersameRhoConstants400PhaseShift180(simulationIndex, totalNumberOfSimulations, swimmerIndex):
        return {
            "oscillationAmplitude": np.pi / 40,
            "oscillationPeriod": 60,  # how many timesteps for one full oscillation
            "oscillationPhaseshift": random.choice([0, np.pi])
        }
    def sameRhoConstants400PhaseShift180(i, numSimulation, defaultTimeSteps):
        return {
            "timeSteps": 7200,
            "timePercentageUsedForMean": 25,

            "environmentSideLength": 10,
            "numSwimmers": 400,
            "interactionRadius": 1,
            "randomAngleAmplitude": 15 * (i / numSimulation),

            "initialVelocity": 0.0025,
            "swimmerSize": 0.04,

            "saveVideo": False,
        }


    # sameRhoGroup400PhaseShift180 = SimulationGroup(simulationDataDir=simulationDir + '/sameRhoGroup400PhaseShift180',
    #                                   constantsFunc=sameRhoConstants400PhaseShift180, initialParameterFunc = initialSwimmerParametersameRhoConstants400PhaseShift180,
    #                                   numSimulation=100, repeatNum=100,
    #                                   saveTrajectoryData=False)
    # simulationManager.appendGroup(sameRhoGroup400PhaseShift180)

    def initialSwimmerParametersameEtaConstantsPhaseShift90(simulationIndex, totalNumberOfSimulations, swimmerIndex):
        return {
            "oscillationAmplitude": np.pi / 40,
            "oscillationPeriod": 60,  # how many timesteps for one full oscillation
            "oscillationPhaseshift": random.choice([0, np.pi / 2])
        }
    def sameEtaConstantsPhaseShift90(i, numSimulation, defaultTimeSteps):
        return {
            "timeSteps": 7200,
            "timePercentageUsedForMean": 25,

            "environmentSideLength": 7,
            "numSwimmers": i + 1,
            "interactionRadius": 1,
            "randomAngleAmplitude": 2,

            "initialVelocity": 0.0025,
            "swimmerSize": 0.04,

            "saveVideo": False,
        }


    # sameEtaGroupPhaseShift90 = SimulationGroup(simulationDataDir=simulationDir + '/sameEtaGroupPhaseShift90', constantsFunc=sameEtaConstantsPhaseShift90,
    #                                 initialParameterFunc = initialSwimmerParametersameEtaConstantsPhaseShift90,
    #                                numSimulation=500, repeatNum=100, saveTrajectoryData=False)
    # simulationManager.appendGroup(sameEtaGroupPhaseShift90)


    def initialSwimmerParametersameEtaConstantsPhaseShift180(simulationIndex, totalNumberOfSimulations, swimmerIndex):
        return {
            "oscillationAmplitude": np.pi / 40,
            "oscillationPeriod": 60,  # how many timesteps for one full oscillation
            "oscillationPhaseshift": random.choice([0, np.pi])
        }
    def sameEtaConstantsPhaseShift180(i, numSimulation, defaultTimeSteps):
        return {
            "timeSteps": 7200,
            "timePercentageUsedForMean": 25,

            "environmentSideLength": 7,
            "numSwimmers": i + 1,
            "interactionRadius": 1,
            "randomAngleAmplitude": 2,

            "initialVelocity": 0.0025,
            "swimmerSize": 0.04,

            "saveVideo": False,
        }


    # sameEtaGroupPhaseShift180 = SimulationGroup(simulationDataDir=simulationDir + '/sameEtaGroupPhaseShift180', constantsFunc=sameEtaConstantsPhaseShift180,
    #                                             initialParameterFunc=initialSwimmerParametersameEtaConstantsPhaseShift180,
    #                                numSimulation=500, repeatNum=100, saveTrajectoryData=False)
    # simulationManager.appendGroup(sameEtaGroupPhaseShift180)

    # simulationManager.simulate()




    def initialSwimmerParameter(simulationIndex, totalNumberOfSimulations, swimmerIndex):
        return {
            "oscillationAmplitude": np.pi / 10,
            "oscillationPeriod": random.choice([30]),  # how many timesteps for one full oscillation
            "oscillationPhaseshift": random.choice([0, np.pi / 2,  np.pi, 3*np.pi/2, 2*np.pi])
        }
    constants = {
        "timeSteps": 1000,
        "timePercentageUsedForMean": 25,

        "environmentSideLength": 5,
        "numSwimmers": 1,
        "interactionRadius": 1,
        "randomAngleAmplitude": 0 * 3,

        "initialVelocity": 0.0025,
        "swimmerSize": 0.04,

        "saveVideo": False,
    }

    simulation = Simulation(constants, initialSwimmerParameter)
    simulation.simulate()
    print(simulation.getAbsoluteVelocityTotal())
    print('That took {} seconds'.format(time.perf_counter() - starttime))
    simulation.animate()

