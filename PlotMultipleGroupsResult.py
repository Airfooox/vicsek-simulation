import os
import json
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors

import util

mpl.use("TkAgg")

if __name__ == "__main__":
    # plotResultDir = r"C:\Users\konst\OneDrive\Uni\Anstellung\Prof. Menzel (2020-22)\vicsek\daten\vicsek-simulation_sameRho_phaseShift"
    # plotSaveFigDir = r"C:\Users\konst\OneDrive\Uni\Anstellung\Prof. Menzel (2020-22)\vicsek\graphen\vicsek-simulation_sameRho_phaseShift"
    plotResultDir = r"C:\Users\konst\OneDrive\Uni\Lehre\7. Semester\Bachelorarbeit\simulationData\simulationData\multiple_groups\two_parameter_sets"
    fileName= r'plotResult_va_over_eta_multiple_groups_two_different_parameters.json'
    file = open(os.path.join(plotResultDir, fileName))

    resultData = json.load(file)
    file.close()

    # attributeAsX = 'interactionStrengthFactor'
    attributeAsX = 'randomAngleAmplitude'

    # print(len(resultData['vectorialGroupVelocities'][0][0]))

    numberOfGroups = len(resultData['groups'][0])
    results = {
        'environmentSideLengths': resultData['environmentSideLengths'],
        'groups': {},
        'vectorialGroupVelocityDifferences': [],
        'x': resultData[attributeAsX],
        'totalAbsoluteVelocity': resultData['totalAbsoluteVelocity'],
        'absoluteVelocityStd': resultData['absoluteVelocityStd'],
    }

    figure = plt.figure()
    gs = figure.add_gridspec(2, hspace=0)
    (ax1, ax2) = gs.subplots(sharex=True)
    figure.set_size_inches(16, 9)

    for scenarioIndex, scenarioGroups in enumerate(resultData['groups']):
        for groupId, group in scenarioGroups.items():
            groupId = int(groupId) - 1
            if groupId not in results['groups'].keys():
                results['groups'][groupId] = {
                    'numSwimmers': group['numSwimmers'],
                    'snakingAmplitude': group['snakingAmplitude'],
                    'snakingPeriod': group['snakingPeriod'],
                    'snakingPhaseshift': group['snakingPhaseshift'],
                    'absoluteVelocity': [],
                    'absoluteVelocityStd': [],
                    'vectorialVelocities': []
                }

            results['groups'][groupId]['absoluteVelocity'].append(resultData['absoluteGroupVelocities'][scenarioIndex][groupId])
            results['groups'][groupId]['absoluteVelocityStd'].append(resultData['absoluteGroupVelocityStd'][scenarioIndex][groupId])
            results['groups'][groupId]['vectorialVelocities'].append([])
            for subScenarioIndex, subScenarioVectorialVelocity in enumerate(resultData['vectorialGroupVelocities'][scenarioIndex]):
                # print(subScenarioIndex, subScenarioVectorialVelocity)
                vectorialVelocity = subScenarioVectorialVelocity[groupId]
                # phi = np.arctan2(vectorialVelocity[1], vectorialVelocity[0])
                results['groups'][groupId]['vectorialVelocities'][scenarioIndex].append(vectorialVelocity)
            # print(scenarioIndex, groupId, results[groupId])

            # plotResultFiles.append(
            #     {'entry': entry, 'x': x, 'y': y, 'std': std})

        results['vectorialGroupVelocityDifferences'].append([])
        for groupId in range(1, numberOfGroups):
            for subScenarioIndex in range(len(results['groups'][groupId]['vectorialVelocities'][scenarioIndex])):
                previousV = results['groups'][groupId - 1]['vectorialVelocities'][scenarioIndex][subScenarioIndex]
                currentV = results['groups'][groupId]['vectorialVelocities'][scenarioIndex][subScenarioIndex]

                dot = previousV[0] * currentV[0] + previousV[1] * currentV[1]
                det = previousV[0] * currentV[1] - previousV[1] * currentV[0]

                # calculate the angle between the inner vectors (so it only goes from 0 to 180°)
                results['vectorialGroupVelocityDifferences'][scenarioIndex].append(np.abs(np.arctan2(det, dot)))


        ax1.scatter([results['x'][scenarioIndex]] * len(results['vectorialGroupVelocityDifferences'][scenarioIndex]), results['vectorialGroupVelocityDifferences'][scenarioIndex], marker='x', s=10)

    ax1.errorbar(results['x'], np.mean(results['vectorialGroupVelocityDifferences'], axis=1), np.std(results['vectorialGroupVelocityDifferences'], axis=1), color='black', marker='', linestyle='-', label='Mean directional difference')

    ax2.errorbar(results['x'], results['totalAbsoluteVelocity'], results['absoluteVelocityStd'], marker='x', linestyle='-', label='Absolute velocity $v_a$')
    for groupId, group in results['groups'].items():
        # print(group)
        ax2.errorbar(results['x'], group['absoluteVelocity'], group['absoluteVelocityStd'], marker='x',
                     linestyle='-', label=f'Absolute velocity $v_{{a, {groupId}}}$ of group {groupId}')
        # ax2.plot(results['x'], group['absoluteVelocity'], marker='',
        #              linestyle='-')

    majorTicks = util.Multiple(8)
    ax1.yaxis.set_major_locator(majorTicks.locator())
    ax1.yaxis.set_major_formatter(majorTicks.formatter())

    # ax1.set_xlabel(r'$\eta \longrightarrow$')
    ax1.set_ylabel(r'$\left| \Delta \theta \right| \longrightarrow$')
    ax2.set_xlabel(r'$\eta \longrightarrow$')
    ax2.set_ylabel(r'$v_a \longrightarrow$')

    ax1.grid('both')
    ax2.grid('both')
    ax1.legend(fontsize=15)
    ax2.legend(fontsize=15)
    # plt.ylim((0,1.01))
    ax1.tick_params(axis='x', labelsize=11 * 1.5)
    ax1.tick_params(axis='y', labelsize=11 * 1.5)
    ax2.tick_params(axis='x', labelsize=11 * 1.5)
    ax2.tick_params(axis='y', labelsize=11 * 1.5)
    ax1.get_xaxis().get_label().set_fontsize(14 * 1.5)
    ax1.get_yaxis().get_label().set_fontsize(14 * 1.5)
    ax2.get_xaxis().get_label().set_fontsize(14 * 1.5)
    ax2.get_yaxis().get_label().set_fontsize(14 * 1.5)

    ax1.set_title(r'Directional difference $\Delta \theta$ as a function of noise strength $\eta$ with different parameter sets $N_1 = N_2 = 1000, \rho = 4, A_1 = \pi/32, T_1 = 150, A_2 = \pi/16, T_2 = 10, \Delta \varphi = 0$')

    plt.show()