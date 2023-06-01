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
    plotResultDir = r"C:\Users\konst\OneDrive\Uni\Lehre\7. Semester\Bachelorarbeit\simulationData\simulationData\multiple_groups"
    # plotResultDir = r"C:\Users\konst\OneDrive\Uni\Lehre\7. Semester\Bachelorarbeit\simulationData\simulationData\multiple_groups\nematic\pi_over_8_and_no_snaking"
    # plotResultDir = r"E:\simulationdata\multiple_groups\nematic\sub\_results"
    fileName= r'plotResult_va_over_eta_multiple_groups_90_phaseshift.json'
    # fileName= r'plotResult_va_over_eta_multiple_groups_961abe6c-fa3c-475e-b670-405aacde3436.json'
    file = open(os.path.join(plotResultDir, fileName))

    resultData = json.load(file)
    file.close()

    # attributeAsX = 'interactionStrengthFactor'
    attributeAsX = 'randomAngleAmplitude'

    # print(len(resultData['vectorialGroupVelocities'][0][0]))

    plotLim = (0, len(resultData[attributeAsX]))
    # plotLim = (0, int((len(resultData[attributeAsX]) - 1)/2) + 1)
    # plotLim = (0, 11)

    numberOfGroups = len(resultData['groups'][0])
    results = {
        'environmentSideLengths': resultData['environmentSideLengths'][plotLim[0]:plotLim[1]+1],
        'groups': {},
        'vectorialGroupVelocityDifferences': resultData['vectorialGroupVelocityDifferences'][plotLim[0]:plotLim[1]+1],
        'vectorialGroupVelocityDifferencesStd': resultData['vectorialGroupVelocityDifferencesStd'][plotLim[0]:plotLim[1]+1],
        'vectorialGroupVelocityAbsDifferences': resultData['vectorialGroupVelocityAbsDifferences'][plotLim[0]:plotLim[1]+1],
        'vectorialGroupVelocityAbsDifferencesStd': resultData['vectorialGroupVelocityAbsDifferencesStd'][plotLim[0]:plotLim[1]+1],
        'x': resultData[attributeAsX][plotLim[0]:plotLim[1]+1],
        'absoluteVelocities': resultData['absoluteVelocities'][plotLim[0]:plotLim[1]+1],
        'totalAbsoluteVelocity': resultData['totalAbsoluteVelocity'][plotLim[0]:plotLim[1]+1],
        'absoluteVelocityStd': resultData['absoluteVelocityStd'][plotLim[0]:plotLim[1]+1],
        'nematicOrderParameters': resultData['nematicOrderParameters'][plotLim[0]:plotLim[1]+1],
        'nematicOrderParameter': resultData['nematicOrderParameter'][plotLim[0]:plotLim[1]+1],
        'nematicOrderParameterStd': resultData['nematicOrderParameterStd'][plotLim[0]:plotLim[1]+1]
    }

    figure = plt.figure(layout='tight')
    subfigures = figure.subfigures(2, 1, height_ratios=[3.0, 1.5])
    gs = subfigures[0].add_gridspec(3, hspace=0)
    (ax1, ax2, ax3) = gs.subplots(sharex=True)
    figure.set_size_inches(16.535, 11.693)
    ax1.set_ylim(-(np.pi+0.25), np.pi+0.25)
    ax2.set_ylim(-0.25, np.pi+0.25)
    gs2 = subfigures[1].add_gridspec(1, 2, wspace=0)
    (ax4, ax5) = gs2.subplots(sharey=True)
    ax4.set_xlim(-(np.pi + 0.25), np.pi + 0.25)
    ax5.set_xlim(-0.25, np.pi + 0.25)

    colors = ['orange', 'green']

    for scenarioIndex, scenarioGroups in enumerate(resultData['groups']):
        if scenarioIndex < plotLim[0] or scenarioIndex > plotLim[1]:
            continue

        for groupId, group in scenarioGroups.items():
            groupId = int(groupId) - 1
            if groupId not in results['groups'].keys():
                results['groups'][groupId] = {
                    'numSwimmers': group['numSwimmers'],
                    'snakingAmplitude': group['snakingAmplitude'],
                    'snakingPeriod': group['snakingPeriod'],
                    'snakingPhaseshift': group['snakingPhaseshift'],
                    'subAbsoluteVelocities': [],
                    'absoluteVelocity': [],
                    'absoluteVelocityStd': [],
                    'vectorialVelocities': [],
                    'nematicOrderParameters': [],
                    'nematicOrderParameter': [],
                    'nematicOrderParameterStd': []
                }

            results['groups'][groupId]['absoluteVelocity'].append(resultData['absoluteGroupVelocities'][scenarioIndex][groupId])
            results['groups'][groupId]['absoluteVelocityStd'].append(resultData['absoluteGroupVelocityStd'][scenarioIndex][groupId])

            results['groups'][groupId]['nematicOrderParameter'].append(resultData['nematicOrderParameterGroups'][scenarioIndex][groupId])
            results['groups'][groupId]['nematicOrderParameterStd'].append(resultData['nematicOrderParameterGroupsStd'][scenarioIndex][groupId])

            results['groups'][groupId]['vectorialVelocities'].append([])
            for subScenarioIndex, subScenarioVectorialVelocity in enumerate(resultData['vectorialGroupVelocities'][scenarioIndex]):

                # print(subScenarioIndex, subScenarioVectorialVelocity)
                vectorialVelocity = subScenarioVectorialVelocity[groupId]
                # phi = np.arctan2(vectorialVelocity[1], vectorialVelocity[0])
                results['groups'][groupId]['vectorialVelocities'][scenarioIndex].append(vectorialVelocity)

            results['groups'][groupId]['subAbsoluteVelocities'].append([])
            for subScenarioIndex, subScenarioAbsoluteVelocities in enumerate(resultData['subAbsoluteGroupVelocities'][scenarioIndex]):
                subAbsoluteVelocity = subScenarioAbsoluteVelocities[groupId]
                results['groups'][groupId]['subAbsoluteVelocities'][scenarioIndex].append(subAbsoluteVelocity)

            results['groups'][groupId]['nematicOrderParameters'].append([])
            for subScenarioIndex, subScenarioNematicOrderParameters in enumerate(resultData['subNematicOrderParametersGroups'][scenarioIndex]):
                nematicOrderParameter = subScenarioNematicOrderParameters[groupId]
                results['groups'][groupId]['nematicOrderParameters'][scenarioIndex].append(nematicOrderParameter)

            # ax3.scatter([results['x'][scenarioIndex]] * len(
            #     results['groups'][groupId]['subAbsoluteVelocities'][scenarioIndex]),
            #             results['groups'][groupId]['subAbsoluteVelocities'][scenarioIndex], marker='x', s=7,
            #             color=colors[groupId])

            ax4.scatter(results['vectorialGroupVelocityDifferences'][scenarioIndex],
                        results['groups'][groupId]['subAbsoluteVelocities'][scenarioIndex], marker='x', s=7,
                        color=colors[groupId])

            ax5.scatter(results['vectorialGroupVelocityAbsDifferences'][scenarioIndex],
                        results['groups'][groupId]['subAbsoluteVelocities'][scenarioIndex], marker='x', s=7,
                        color=colors[groupId])

            # ax3.scatter([results['x'][scenarioIndex]] * len(
            #     results['groups'][groupId]['nematicOrderParameters'][scenarioIndex]),
            #             results['groups'][groupId]['nematicOrderParameters'][scenarioIndex], marker='x', s=7,
            #             color=colors[groupId])
            #
            # ax4.scatter(results['vectorialGroupVelocityDifferences'][scenarioIndex],
            #             results['groups'][groupId]['nematicOrderParameters'][scenarioIndex], marker='x', s=7,
            #             color=colors[groupId])

        ax1.scatter([results['x'][scenarioIndex]] * len(results['vectorialGroupVelocityDifferences'][scenarioIndex]), results['vectorialGroupVelocityDifferences'][scenarioIndex], marker='x', s=7, color='black')
        ax2.scatter([results['x'][scenarioIndex]] * len(results['vectorialGroupVelocityAbsDifferences'][scenarioIndex]), results['vectorialGroupVelocityAbsDifferences'][scenarioIndex], marker='x', s=7, color='black')
        # ax3.scatter([results['x'][scenarioIndex]] * len(results['absoluteVelocities'][scenarioIndex]), results['absoluteVelocities'][scenarioIndex], marker='x', s=7, color='black')
        # ax3.scatter([results['x'][scenarioIndex]] * len(results['nematicOrderParameters'][scenarioIndex]), results['nematicOrderParameters'][scenarioIndex], marker='x', s=7, color='black')

        ax4.scatter(results['vectorialGroupVelocityDifferences'][scenarioIndex], results['absoluteVelocities'][scenarioIndex], marker='x', s=7, color='black')
        ax5.scatter(results['vectorialGroupVelocityAbsDifferences'][scenarioIndex], results['absoluteVelocities'][scenarioIndex], marker='x', s=7, color='black')
        # ax4.scatter(results['vectorialGroupVelocityDifferences'][scenarioIndex], results['nematicOrderParameters'][scenarioIndex], marker='x', s=7, color='black')

    # print(len(results['x']), len(np.std(results['vectorialGroupVelocityDifferences'], axis=1)))
    # https://stats.stackexchange.com/questions/168971/variance-of-an-average-of-random-variables
    vectorialGroupVelocityDifferencesStd = []
    for listOfVelocityDifferences, listOfStandardDeviations in zip(results['vectorialGroupVelocityDifferences'], results['vectorialGroupVelocityDifferencesStd']):
        listOfVelocityDifferences = np.array(listOfVelocityDifferences)[:, 0]
        listOfStandardDeviations = np.array(listOfStandardDeviations)[:, 0]
        listOfVariances = np.array(listOfStandardDeviations) ** 2

        variance = np.sum(listOfVariances) / len(listOfVariances)**2
        std = np.sqrt(variance)
        vectorialGroupVelocityDifferencesStd.append(std)

        # https://stats.stackexchange.com/questions/168971/variance-of-an-average-of-random-variables
        vectorialGroupVelocityAbsDifferencesStd = []
        for listOfVelocityDifferences, listOfStandardDeviations in zip(results['vectorialGroupVelocityAbsDifferences'],
                                                                       results['vectorialGroupVelocityAbsDifferencesStd']):
            listOfVelocityDifferences = np.array(listOfVelocityDifferences)[:, 0]
            listOfStandardDeviations = np.array(listOfStandardDeviations)[:, 0]
            listOfVariances = np.array(listOfStandardDeviations) ** 2

            variance = np.sum(listOfVariances) / len(listOfVariances) ** 2
            std = np.sqrt(variance)
            vectorialGroupVelocityAbsDifferencesStd.append(std)

    # ax1.errorbar(results['x'], np.mean(results['vectorialGroupVelocityDifferences'], axis=1)[:, 0], vectorialGroupVelocityDifferencesStd, color='red', marker='x', linestyle='-', label='Mean directional difference')
    ax1.errorbar(results['x'], np.mean(results['vectorialGroupVelocityDifferences'], axis=1)[:, 0], np.std(results['vectorialGroupVelocityDifferences'], axis=1)[:, 0], color='red', marker='x', linestyle='-', label='Mean directional difference')
    ax2.errorbar(results['x'], np.mean(results['vectorialGroupVelocityAbsDifferences'], axis=1)[:, 0], vectorialGroupVelocityAbsDifferencesStd, color='red', marker='x', linestyle='-', label='Mean absolute directional difference')

    ax3.errorbar(results['x'], results['totalAbsoluteVelocity'], results['absoluteVelocityStd'], marker='x', linestyle='-', color='black', label='Absolute velocity $v_a$')
    # ax3.errorbar(results['x'], results['nematicOrderParameter'], results['nematicOrderParameterStd'], marker='x', linestyle='-', label='Nematic order parameter $S$')

    for groupId, group in results['groups'].items():
        # print(group)
        ax3.errorbar(results['x'], group['absoluteVelocity'], group['absoluteVelocityStd'], marker='x',
                     linestyle='-', label=f'Absolute velocity $v_{{a, {groupId + 1}}}$ of group {groupId + 1}', color=colors[groupId])
        # ax3.errorbar(results['x'], group['nematicOrderParameter'], group['nematicOrderParameterStd'], marker='x',
        #              linestyle='-', label=f'Nematic order parameter $S_{{{groupId + 1}}}$', color=colors[groupId])
        # ax3.plot(results['x'], group['absoluteVelocity'], marker='',
        #              linestyle='-')

    majorTicks1 = util.Multiple(2)
    ax1.yaxis.set_major_locator(majorTicks1.locator())
    ax1.yaxis.set_major_formatter(majorTicks1.formatter())
    majorTicks2= util.Multiple(4)
    ax2.yaxis.set_major_locator(majorTicks2.locator())
    ax2.yaxis.set_major_formatter(majorTicks2.formatter())

    ax4.xaxis.set_major_locator(majorTicks1.locator())
    ax4.xaxis.set_major_formatter(majorTicks1.formatter())
    ax5.xaxis.set_major_locator(majorTicks2.locator())
    ax5.xaxis.set_major_formatter(majorTicks2.formatter())

    # ax1.set_xlabel(r'$\eta \longrightarrow$')
    ax1.set_ylabel(r'$\overline{\Delta \theta} \longrightarrow$')
    ax2.set_ylabel(r'$\overline{\left| \Delta \theta \right|} \longrightarrow$')
    ax3.set_xlabel(r'$\eta \longrightarrow$')
    ax3.set_ylabel(r'$v_a \longrightarrow$')
    # ax3.set_ylabel(r'$S \longrightarrow$')

    # ax4.set_xlabel(r'$\left| \Delta \theta \right| \longrightarrow$')
    ax4.set_xlabel(r'$\overline{\Delta \theta}  \longrightarrow$')
    ax4.set_ylabel(r'$v_a \longrightarrow$')
    ax5.set_xlabel(r'$\overline{\left| \Delta \theta \right|} \longrightarrow$')
    # ax4.set_ylabel(r'$S \longrightarrow$')

    ax1.grid('both')
    ax2.grid('both')
    ax3.grid('both')
    ax4.grid('both')
    ax5.grid('both')
    # plt.ylim((0,1.01))
    ax1.legend(fontsize=15)
    ax2.legend(fontsize=15)
    ax3.legend(fontsize=15)
    ax4.legend(fontsize=15)
    ax5.legend(fontsize=15)
    ax1.tick_params(axis='x', labelsize=11 * 1.5)
    ax1.tick_params(axis='y', labelsize=11 * 1.5)
    ax2.tick_params(axis='x', labelsize=11 * 1.5)
    ax2.tick_params(axis='y', labelsize=11 * 1.5)
    ax3.tick_params(axis='x', labelsize=11 * 1.5)
    ax3.tick_params(axis='y', labelsize=11 * 1.5)
    ax4.tick_params(axis='x', labelsize=11 * 1.5)
    ax4.tick_params(axis='y', labelsize=11 * 1.5)
    ax5.tick_params(axis='x', labelsize=11 * 1.5)
    ax5.tick_params(axis='y', labelsize=11 * 1.5)
    ax1.get_xaxis().get_label().set_fontsize(14 * 1.5)
    ax1.get_yaxis().get_label().set_fontsize(14 * 1.5)
    ax2.get_xaxis().get_label().set_fontsize(14 * 1.5)
    ax2.get_yaxis().get_label().set_fontsize(14 * 1.5)
    ax3.get_xaxis().get_label().set_fontsize(14 * 1.5)
    ax3.get_yaxis().get_label().set_fontsize(14 * 1.5)
    ax4.get_xaxis().get_label().set_fontsize(14 * 1.5)
    ax4.get_yaxis().get_label().set_fontsize(14 * 1.5)
    ax5.get_xaxis().get_label().set_fontsize(14 * 1.5)
    ax5.get_yaxis().get_label().set_fontsize(14 * 1.5)

    # ax1.set_title(r'Directional difference $\Delta \theta$ as a function of noise strength $\eta$ with 90° phaseshift $N_1 = N_2 = 1000, \rho = 4, A_1 = A_2 = \pi/16, T_1 = T_2 = 30, \Delta \varphi = \pi/2$')

    # figure.tight_layout()
    figure.savefig(os.path.join(plotResultDir, fileName[:-5] + ".png"), bbox_inches='tight')
    plt.show()