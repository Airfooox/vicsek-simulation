# setup
import random
import os
import uuid

import util
from Simulation import Simulation
from util import printProgressBar
import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("TkAgg")

if __name__ == "__main__":
    starttime = time.perf_counter()
    N = 2000
    rho = 4
    # N = int(512**2 * rho)

    simulationConfig = {
        "timeSteps": 7500,

        "environmentSideLength": np.sqrt(N / rho),
        "groups": {
            '1': {
                'numSwimmers': 1000,
                'snakingAmplitude': np.pi/16,
                'snakingPeriod': 30,
                'snakingPhaseshift': 0
            },
            '2': {
                'numSwimmers': 1000,
                'snakingAmplitude': np.pi/16,
                'snakingPeriod': 30,
                'snakingPhaseshift': np.pi
            },
        },
        "interactionRadius": 1,
        "randomAngleAmplitude": 0.35, # 0.35
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

    absoluteVelocities = simulation.absoluteVelocities
    absoluteGroupVelocities = simulation.absoluteGroupVelocities
    vectorialGroupVelocities = simulation.vectorialGroupVelocities
    numberOfGroups = len(simulation.simulationConfig['groups'])
    vectorialGroupDifferences = []
    vectorialGroupAbsDifferences = []
    groupDirections = [[], []]
    for groupId in range(numberOfGroups - 1):
        for t in range(simulation.simulationConfig['timeSteps']):
            previousV = vectorialGroupVelocities[t, groupId]
            currentV = vectorialGroupVelocities[t, groupId + 1]

            ang1 = np.arctan2(previousV[1], previousV[0]) % (2*np.pi)
            ang2 = np.arctan2(currentV[1], currentV[0]) % (2*np.pi)
            groupDirections[groupId].append(ang1)
            groupDirections[groupId + 1].append(ang2)
            # angle = (ang1 - ang2) % (2 * np.pi)
            # innerAngle = np.abs(angle)

            dot = previousV[0] * currentV[0] + previousV[1] * currentV[1]
            det = previousV[0] * currentV[1] - previousV[1] * currentV[0]

            # calculate the angle between the inner vectors (so it only goes from 0 to 180°)
            angle = np.arctan2(det, dot)
            innerAngle = np.abs(np.arctan2(det, dot))
            vectorialGroupDifferences.append(angle)
            vectorialGroupAbsDifferences.append(innerAngle)

    figure = plt.figure()
    gs = figure.add_gridspec(4, hspace=0)
    (ax1, ax2, ax3, ax4) = gs.subplots(sharex=True)
    figure.set_size_inches(11.693, 8.268)
    ax1.plot(range(simulation.simulationConfig['timeSteps']), vectorialGroupDifferences)
    ax2.plot(range(simulation.simulationConfig['timeSteps']), vectorialGroupAbsDifferences)
    ax3.plot(range(simulation.simulationConfig['timeSteps']), groupDirections[0], label=r'$\theta_1(t)$')
    ax3.plot(range(simulation.simulationConfig['timeSteps']), groupDirections[1], label=r'$\theta_2(t)$')
    ax4.plot(range(simulation.simulationConfig['timeSteps']), absoluteVelocities, label=r'$v_a$')
    ax4.plot(range(simulation.simulationConfig['timeSteps']), absoluteGroupVelocities[:, 0], label=r'$v_{a, 1}$')
    ax4.plot(range(simulation.simulationConfig['timeSteps']), absoluteGroupVelocities[:, 1], label=r'$v_{a, 2}$')
    meanAx1, stdAx1 = np.mean(vectorialGroupDifferences[simulation.framesUsedForMean:]), np.std(vectorialGroupDifferences[simulation.framesUsedForMean:])
    meanAx2, stdAx2 = np.mean(vectorialGroupAbsDifferences[simulation.framesUsedForMean:]), np.std(vectorialGroupAbsDifferences[simulation.framesUsedForMean:])

    ax1.axhline(y=meanAx1, color='r', linestyle='-', label=r'$\overline{\Delta \theta} = \left\langle \Delta \theta(t) \right\rangle_{t>T}$')
    ax1.axhline(y=meanAx1 + stdAx1, color='r', linestyle='--')
    ax1.axhline(y=meanAx1 - stdAx1, color='r', linestyle='--')

    ax2.axhline(y=meanAx2, color='r', linestyle='-', label=r'$\overline{\left| \Delta \theta \right|} = \left\langle \left| \Delta \theta \right| (t) \right\rangle_{t>T} $')
    ax2.axhline(y=meanAx2 + stdAx2, color='r', linestyle='--')
    ax2.axhline(y=meanAx2 - stdAx2, color='r', linestyle='--')

    ymin, ymax = ax1.get_ylim()
    ax1.annotate('$T$', xy=(simulation.framesUsedForMean, ymax), xytext=(-3, 3), textcoords='offset points', annotation_clip=False, fontsize=13*1.5)
    ax1.axvline(simulation.framesUsedForMean, 0, 1, color="b")
    ax2.axvline(simulation.framesUsedForMean, 0, 1, color="b")
    ax3.axvline(simulation.framesUsedForMean, 0, 1, color="b")
    ax4.axvline(simulation.framesUsedForMean, 0, 1, color="b")
    ax1.set_title(
        rf'$\overline{{\Delta \theta}} = ({np.round(meanAx1, 4)} \pm {np.round(stdAx1, 4)}), $' +
        rf'$\overline{{\left| \Delta \theta \right|}} = ({np.round(meanAx2, 4)} \pm {np.round(stdAx2, 4)})$',
        fontsize=13
    )

    # ax1.set_ylim(-(np.pi + 0.25), np.pi + 0.25)
    # ax2.set_ylim(-0.25, np.pi + 0.25)
    # ax3.set_ylim(-0.25, 2*np.pi + 0.25)
    majorTicks = util.Multiple(8)
    ax1.yaxis.set_major_locator(majorTicks.locator())
    ax1.yaxis.set_major_formatter(majorTicks.formatter())
    ax2.yaxis.set_major_locator(majorTicks.locator())
    ax2.yaxis.set_major_formatter(majorTicks.formatter())
    majorTicks3 = util.Multiple(4)
    ax3.yaxis.set_major_locator(majorTicks3.locator())
    ax3.yaxis.set_major_formatter(majorTicks3.formatter())

    ax1.set_ylabel(r'$\Delta \theta(t) \longrightarrow$')
    ax2.set_ylabel(r'$\left| \Delta \theta (t) \right| \longrightarrow$')
    ax3.set_ylabel(r'$\theta_i (t) \longrightarrow$')
    ax4.set_ylabel(r'$v_a (t) \longrightarrow$')
    ax4.set_xlabel(r'$t \longrightarrow$')
    ax1.grid('both')
    ax2.grid('both')
    ax3.grid('both')
    ax4.grid('both')
    ax1.tick_params(axis='x', labelsize=11 * 1.5)
    ax1.tick_params(axis='y', labelsize=11 * 1.5)
    ax2.tick_params(axis='x', labelsize=11 * 1.5)
    ax2.tick_params(axis='y', labelsize=11 * 1.5)
    ax3.tick_params(axis='x', labelsize=11 * 1.5)
    ax3.tick_params(axis='y', labelsize=11 * 1.5)
    ax4.tick_params(axis='x', labelsize=11 * 1.5)
    ax4.tick_params(axis='y', labelsize=11 * 1.5)
    ax1.get_xaxis().get_label().set_fontsize(14 * 1.5)
    ax1.get_yaxis().get_label().set_fontsize(14 * 1.5)
    ax2.get_xaxis().get_label().set_fontsize(14 * 1.5)
    ax2.get_yaxis().get_label().set_fontsize(14 * 1.5)
    ax3.get_xaxis().get_label().set_fontsize(14 * 1.5)
    ax3.get_yaxis().get_label().set_fontsize(14 * 1.5)
    ax4.get_xaxis().get_label().set_fontsize(14 * 1.5)
    ax4.get_yaxis().get_label().set_fontsize(14 * 1.5)
    ax1.legend(fontsize=14)
    ax2.legend(fontsize=14)
    ax3.legend(fontsize=14)
    ax4.legend(fontsize=14)


    print('mean/std 1:', meanAx1, stdAx1)
    print('mean/std 2:', meanAx2, stdAx2)
    plt.show()
    starttime = time.perf_counter()
    print('Absolute velocity took {} seconds'.format(time.perf_counter() - starttime))

    simulation.animate(showGroup=False)
    while True:
        showGroupAnimation = input('Watch the version with the groups painted? (y/n)')
        if showGroupAnimation.lower() == 'y':
            simulation.animate(showGroup=True)
            while True:
                saveFiles = input('Save both video files? (y/n)')
                if saveFiles.lower() == 'y':
                    amplitudes = ', '.join([str(np.round(groupData['snakingAmplitude'], 3)) + 'pi' for groupData in simulationConfig['groups'].values()])
                    periods = ', '.join([str(groupData['snakingPeriod']) for groupData in simulationConfig['groups'].values()])
                    phaseShifts = ', '.join([str(np.round(groupData['snakingPhaseshift'], 3)) + 'pi' for groupData in simulationConfig['groups'].values()])

                    # videoPath = r'C:\Users\konst\OneDrive\Uni\Anstellung\Prof. Menzel (2020-22)\vicsek\simulation\videos'
                    videoPath = r'E:\simulationdata\multiple_groups'
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