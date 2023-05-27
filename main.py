# setup
import random
import os
import uuid

from Simulation import Simulation
from util import printProgressBar
import numpy as np
import time
import matplotlib.pyplot as plt

if __name__ == "__main__":
    starttime = time.perf_counter()
    N = 2000
    rho = 4
    # N = int(512**2 * rho)

    simulationConfig = {
        "timeSteps": 5000,

        "environmentSideLength": np.sqrt(N / rho),
        "groups": {
            '1': {
                'numSwimmers': 1000,
                'snakingAmplitude': np.pi/8,
                'snakingPeriod': 30,
                'snakingPhaseshift': 0
            },
            '2': {
                'numSwimmers': 1000,
                'snakingAmplitude': 0,
                'snakingPeriod': 30,
                'snakingPhaseshift': 0
            }
        },
        "interactionRadius": 1,
        "randomAngleAmplitude": 0.5,
        "interactionStrengthFactor": 0.15,

        "velocity": 0.0025,

        "saveVideo": False,
    }

    simulation = Simulation(simulationIndex=1, numSimulation=1, simulationConfig=simulationConfig, timePercentageUsedForMean=25)
    print('Setup took {} seconds'.format(time.perf_counter() - starttime))
    starttime = time.perf_counter()
    simulation.simulate()
    print('totalAbsoluteVelocity', simulation.totalAbsoluteVelocity)
    print('totalAbsoluteGroupVelocities', simulation.totalAbsoluteGroupVelocities)
    print('totalVectorialVelocity', simulation.totalVectorialVelocity)
    print('totalVectorialGroupVelocities', simulation.totalVectorialGroupVelocities)
    print('totalNematicOrderParameter', simulation.totalNematicOrderParameter)
    print('totalNematicOrderParameterGroups', simulation.totalNematicOrderParameterGroups)
    print('Simulation took {} seconds'.format(time.perf_counter() - starttime))

    vectorialGroupVelocities = simulation.vectorialGroupVelocities
    numberOfGroups = len(simulation.simulationConfig['groups'])
    vectorialGroupDifferences = []
    for groupId in range(numberOfGroups - 1):
        for t in range(simulation.simulationConfig['timeSteps']):
            previousV = vectorialGroupVelocities[t, groupId]
            currentV = vectorialGroupVelocities[t, groupId + 1]

            ang1 = np.arctan2(previousV[1], previousV[0])
            ang2 = np.arctan2(currentV[1], currentV[0])
            angle = (ang1 - ang2) % (2 * np.pi)

            # dot = previousV[0] * currentV[0] + previousV[1] * currentV[1]
            # det = previousV[0] * currentV[1] - previousV[1] * currentV[0]
            #
            # # calculate the angle between the inner vectors (so it only goes from 0 to 180°)
            # angl = np.abs(np.arctan2(det, dot))
            vectorialGroupDifferences.append(angle)

    plt.plot(range(simulation.simulationConfig['timeSteps']), vectorialGroupDifferences)
    mean = np.mean(vectorialGroupDifferences[simulation.framesUsedForMean:])
    std = np.std(vectorialGroupDifferences[simulation.framesUsedForMean:])
    meanLine = plt.axhline(y=mean, color='r', linestyle='-')
    meanOStdLine = plt.axhline(y=mean + std, color='r', linestyle='--')
    meanUStdLine = plt.axhline(y=mean - std, color='r', linestyle='--')
    print('mean/std:', mean, std)
    plt.show()
    # starttime = time.perf_counter()
    # print('Absolute velocity took {} seconds'.format(time.perf_counter() - starttime))

    # simulation.animate(showGroup=False)
    # while True:
    #     showGroupAnimation = input('Watch the version with the groups painted? (y/n)')
    #     if showGroupAnimation.lower() == 'y':
    #         simulation.animate(showGroup=True)
    #         while True:
    #             saveFiles = input('Save both video files? (y/n)')
    #             if saveFiles.lower() == 'y':
    #                 amplitudes = ', '.join([str(np.round(groupData['snakingAmplitude'], 3)) + 'pi' for groupData in simulationConfig['groups'].values()])
    #                 periods = ', '.join([str(groupData['snakingPeriod']) for groupData in simulationConfig['groups'].values()])
    #                 phaseShifts = ', '.join([str(np.round(groupData['snakingPhaseshift'], 3)) + 'pi' for groupData in simulationConfig['groups'].values()])
    #
    #                 # videoPath = r'C:\Users\konst\OneDrive\Uni\Anstellung\Prof. Menzel (2020-22)\vicsek\simulation\videos'
    #                 videoPath = r'E:\simulationdata\multiple_groups'
    #                 uuid = uuid.uuid4()
    #                 # noGroupVideoName = f'singleSwimmerTrajectory_t={simulationConfig["timeSteps"]}_N={N}_r={rho}_e={simulationConfig["randomAngleAmplitude"]}_a=[{amplitudes}]_T=[{periods}]_phS=[{phaseShifts}]_{uuid}.mp4'
    #                 noGroupVideoName = f'singleSwimmerTrajectory_{uuid}.mp4'
    #                 # groupVideoName = f'singleSwimmerTrajectory_t={simulationConfig["timeSteps"]}_N={N}_r={rho}_e={simulationConfig["randomAngleAmplitude"]}_a=[{amplitudes}]_T=[{periods}]_phS=[{phaseShifts}]_group_{uuid}.mp4'
    #                 groupVideoName = f'singleSwimmerTrajectory_group_{uuid}.mp4'
    #                 simulation.animate(showGroup=False, saveVideo=True, videoPath=os.path.join(videoPath, noGroupVideoName))
    #                 simulation.animate(showGroup=True, saveVideo=True, videoPath=os.path.join(videoPath, groupVideoName))
    #                 break
    #             if saveFiles.lower() == 'n':
    #                 break
    #
    #             print(f'Invalid input: {saveFiles}')
    #
    #         if saveFiles == 'y' or 'n':
    #             break
    #     if showGroupAnimation.lower() == 'n':
    #         break
    #
    #     print(f'Invalid input: {showGroupAnimation}')