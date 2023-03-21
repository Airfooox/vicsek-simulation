# setup
import random
import os
import uuid

from Simulation import Simulation
from util import printProgressBar
import numpy as np
import time

if __name__ == "__main__":
    starttime = time.perf_counter()
    N = 1000
    rho = 4
    # N = int(512**2 * rho)

    simulationConfig = {
        "timeSteps": 2500,

        "environmentSideLength": np.sqrt(N / rho),
        "groups": {
            "1": {
                "numSwimmers": N,
                "oscillationAmplitude": np.pi/16 * 0,
                "oscillationPeriod": 30,  # how many timesteps for one full oscillation
                "oscillationPhaseShift": 0
            },
            # "2": {
            #     "numSwimmers": 200,
            #     "oscillationAmplitude":  np.pi / 32,
            #     "oscillationPeriod": 100,  # how many timesteps for one full oscillation
            #     "oscillationPhaseShift": np.pi / 2
            # }
        },
        "interactionRadius": 1,
        "randomAngleAmplitude": 0,
        "interactionStrengthFactor": 0.5,

        "velocity": 0.0025,

        "saveVideo": False,
    }

    simulation = Simulation(simulationIndex=1, numSimulation=1, simulationConfig=simulationConfig, timePercentageUsedForMean=25)
    print('Setup took {} seconds'.format(time.perf_counter() - starttime))
    starttime = time.perf_counter()
    simulation.simulate()
    print(simulation.totalAbsoluteVelocity)
    print(simulation.totalAbsoluteGroupVelocities)
    print(simulation.totalVectorialVelocity)
    print(simulation.totalVectorialGroupVelocities)
    print('Simulation took {} seconds'.format(time.perf_counter() - starttime))
    # starttime = time.perf_counter()
    # print('Absolute velocity took {} seconds'.format(time.perf_counter() - starttime))

    simulation.animate(showGroup=False)
    while True:
        showGroupAnimation = input('Watch the version with the groups painted? (y/n)')
        if showGroupAnimation.lower() == 'y':
            simulation.animate(showGroup=True)
            while True:
                saveFiles = input('Save both video files? (y/n)')
                if saveFiles.lower() == 'y':
                    amplitudes = ', '.join([str(np.round(groupData['oscillationAmplitude'], 3)) + 'pi' for groupData in simulationConfig['groups'].values()])
                    periods = ', '.join([str(groupData['oscillationPeriod']) for groupData in simulationConfig['groups'].values()])
                    phaseShifts = ', '.join([str(np.round(groupData['oscillationPhaseShift'], 3)) + 'pi' for groupData in simulationConfig['groups'].values()])

                    # videoPath = r'C:\Users\konst\OneDrive\Uni\Anstellung\Prof. Menzel (2020-22)\vicsek\simulation\videos'
                    videoPath = r'D:\simulationdata\_videos'
                    uuid = uuid.uuid4()
                    # noGroupVideoName = f'singleSwimmerTrajectory_t={simulationConfig["timeSteps"]}_N={N}_r={rho}_e={simulationConfig["randomAngleAmplitude"]}_a=[{amplitudes}]_T=[{periods}]_phS=[{phaseShifts}]_{uuid}.mp4'
                    noGroupVideoName = f'singleSwimmerTrajectory_{uuid}.mp4'
                    # groupVideoName = f'singleSwimmerTrajectory_t={simulationConfig["timeSteps"]}_N={N}_r={rho}_e={simulationConfig["randomAngleAmplitude"]}_a=[{amplitudes}]_T=[{periods}]_phS=[{phaseShifts}]_group_{uuid}.mp4'
                    groupVideoName = f'singleSwimmerTrajectory_group_{uuid}.mp4'
                    simulation.animate(showGroup=False, saveVideo=True, videoPath=os.path.join(videoPath, noGroupVideoName))
                    simulation.animate(showGroup=True, saveVideo=True, videoPath=os.path.join(videoPath, groupVideoName))
                    break
                if saveFiles.lower() == 'n':
                    break

                print(f'Invalid input: {saveFiles}')

            if saveFiles == 'y' or 'n':
                break
        if showGroupAnimation.lower() == 'n':
            break

        print(f'Invalid input: {showGroupAnimation}')