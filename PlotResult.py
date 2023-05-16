import os
import json
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
mpl.use("TkAgg")

if __name__ == "__main__":
    # plotResultDir = r"C:\Users\konst\OneDrive\Uni\Anstellung\Prof. Menzel (2020-22)\vicsek\daten\vicsek-simulation_sameRho_phaseShift"
    # plotSaveFigDir = r"C:\Users\konst\OneDrive\Uni\Anstellung\Prof. Menzel (2020-22)\vicsek\graphen\vicsek-simulation_sameRho_phaseShift"
    plotResultDir = r"C:\Users\konst\OneDrive\Uni\Lehre\7. Semester\Bachelorarbeit\simulationData\simulationData\multiple_groups\two_periods"
    # plotSaveFigDir = r"C:\Users\konst\OneDrive\Uni\Lehre\7. Semester\Bachelorarbeit\simulationData\simulationData\va_over_eta_N=1000_rho=4_A=0.06pi_T=30_g=1 or 0.1\_results"
    plotSaveFigDir = plotResultDir
    dirList = os.listdir(plotResultDir)

    # attributeAsX = 'interactionStrengthFactor'
    attributeAsX = 'randomAngleAmplitude'
    plotNumSwimmersWhitelist = []
    plotRhoWhitelist = []
    plotEtaWhitelist = []
    plotAmplitudeWhitelist = []
    plotPeriodWhitelist = []
    plotInteractionStrengthWhitelist = []
    applyNumSwimmersWhitelist = len(plotNumSwimmersWhitelist) != 0
    applyRhoWhitelist = len(plotRhoWhitelist) != 0
    applyEtaWhitelist = len(plotEtaWhitelist) != 0
    applyAmplitudeWhitelist = len(plotAmplitudeWhitelist) != 0
    applyPeriodWhitelist = len(plotPeriodWhitelist) != 0
    applyInteractionStrengthWhitelist = len(plotInteractionStrengthWhitelist) != 0

    plotResultFiles = []
    for entry in dirList:
        filePath = os.path.join(plotResultDir, entry)
        prefixLength = 0
        # remove .png from end and split it
        entryArgs = entry[:-4].split('_')[prefixLength:]
        # print(entryArgs)
        if os.path.isfile(filePath) and len(entryArgs) > 1 and entry.split('_')[0] == 'plotResult':
            # numSwimmers = entryArgs[0].split('=')[1]
            # rho = entryArgs[1].split('=')[1]
            # interactionStrength = entryArgs[2].split('=')[1]
            # amplitude = entryArgs[3].split('=')[1]
            # period = entryArgs[4].split('=')[1]
            # phaseShift = entryArgs[5].split('=')[1]

            # numSwimmers = entryArgs[0].split('=')[1]
            # rho = entryArgs[1].split('=')[1]
            # eta = entryArgs[2].split('=')[1]
            # amplitude = entryArgs[3].split('=')[1]
            # period = entryArgs[4].split('=')[1]
            # phaseShift = entryArgs[5].split('=')[1]
            #
            # if applyNumSwimmersWhitelist and numSwimmers not in plotNumSwimmersWhitelist:
            #     continue
            #
            # if applyRhoWhitelist and rho not in plotRhoWhitelist:
            #     continue
            #
            # if applyEtaWhitelist and eta not in plotEtaWhitelist:
            #     continue
            #
            # if applyAmplitudeWhitelist and amplitude not in plotAmplitudeWhitelist:
            #     continue
            #
            # if applyPeriodWhitelist and period not in plotPeriodWhitelist:
            #     continue

            # if applyInteractionStrengthWhitelist and interactionStrength not in plotInteractionStrengthWhitelist:
            #     continue

            with open(filePath) as plotFile:
                resultObj = json.load(plotFile)
                x = resultObj[attributeAsX]
                # y = np.array(resultObj['totalAbsoluteVelocity'])
                y = np.array(resultObj['absoluteGroupVelocities'])[:, 0]
                std = resultObj['absoluteVelocityStd']

            plotResultFiles.append(
                {'entry': entry, 'x': x, 'y': y, 'std': std})

    colors = cm.tab20(range(len(plotResultFiles)))

    for i, plotResultEntry in enumerate(plotResultFiles):
        # numSwimmers = plotResultEntry['numSwimmers']
        # rho = plotResultEntry['rho']
        # eta = plotResultEntry['eta']
        # amplitude = plotResultEntry['amplitude']
        # period = plotResultEntry['period']
        # phaseShift = plotResultEntry['phaseShift']
        # if int(numSwimmers) == 400:
        #     marker='^'
        # elif int(numSwimmers) == 1000:
        #     marker='x'
        # else:
        #     marker='D'

        # if float(amplitude[:-2]) == 0:
        #     amplitude = '0'
        # elif amplitude[-2:] == "pi":
        #     amplitude = amplitude[:-2] + "\pi"



        # label = '$'
        # # label += rf'N={numSwimmers}, '
        # # label += rf'\varrho={rho}, '
        # # label += rf'g={interactionStrength}, '
        # label += rf'\eta={eta}, '
        # label += rf'A={amplitude}, '
        # if amplitude != '0':
        #     label += rf'T={period}, '
        # # label += rf'\Delta \varphi={phaseShift} '
        # label = label[:-2]
        # label += '$'
        # plt.errorbar(plotResultEntry['x'], plotResultEntry['y'], yerr=plotResultEntry['std'], fmt= matplotlib.colors.rgb2hex(colors[i]), marker=marker, linestyle='-',
        #              label=label)
        plt.errorbar(plotResultEntry['x'], plotResultEntry['y'], yerr=plotResultEntry['std'], marker='x', linestyle='',
                     fmt=matplotlib.colors.rgb2hex(colors[i]))

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(16, 9)

    # plt.xlabel(r'$g \longrightarrow$')
    plt.xlabel(r'$\eta \longrightarrow$')
    # plt.xlabel(r'$A \longrightarrow$')
    plt.ylabel(r'$v_a \longrightarrow$')
    plt.grid('both')
    plt.legend(fontsize=15)
    # plt.ylim((0,1.01))
    plt.xticks(fontsize=11 * 1.5)
    plt.yticks(fontsize=11 * 1.5)
    figure.axes[0].get_xaxis().get_label().set_fontsize(14 * 1.5)
    figure.axes[0].get_yaxis().get_label().set_fontsize(14 * 1.5)


    numSwimmersStr = ','.join(f'{x}' for x in plotNumSwimmersWhitelist)
    rhoStr = ','.join(f'{x}' for x in plotRhoWhitelist)
    amplitudeStr = ','.join(f'{x}' for x in plotAmplitudeWhitelist)
    periodStr = ','.join(f'{x}' for x in plotPeriodWhitelist)
    plotSaveFigName = f"plot_numSwimmers=[{numSwimmersStr}]_rho=[{rhoStr}]_amplitude=[{amplitudeStr}]_period=[{periodStr}].png"
    plt.savefig(os.path.join(plotSaveFigDir, plotSaveFigName), bbox_inches='tight')
    plt.show()

    # timeEvolutionFile = 'D:/simulationdata/plotResult.txt'
    # simulationScenarioNum = 48
    #
    # with open(timeEvolutionFile) as plotFile:
    #     timeEvolution = json.load(plotFile)
    #
    # y = timeEvolution[simulationScenarioNum]
    # x = range(len(y))
    #
    # plt.plot(x, y)
    # plt.show()
