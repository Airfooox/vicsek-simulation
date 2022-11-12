import os
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors

if __name__ == "__main__":
    plotResultDir = "C:\\Users\\konst\\OneDrive\\Uni\\Anstellung\\Prof. Menzel (2020-22)\\vicsek\\daten\\vicsek-simulation_sameRho_phaseShift"
    plotSaveFigDir = "C:\\Users\\konst\\OneDrive\\Uni\\Anstellung\\Prof. Menzel (2020-22)\\vicsek\\graphen\\vicsek-simulation_sameRho_phaseShift"
    dirList = os.listdir(plotResultDir)

    attributeAsX = 'randomAngleAmplitude'
    plotNumSwimmersWhitelist = ["1000"]
    plotRhoWhitelist = ["2"]
    plotAmplitudeWhitelist = ["0.02pi", "0.03pi", "0.06pi"]
    plotPeriodWhitelist = ["50"]
    applyNumSwimmersWhitelist = len(plotNumSwimmersWhitelist) != 0
    applyRhoWhitelist = len(plotRhoWhitelist) != 0
    applyAmplitudeWhitelist = len(plotAmplitudeWhitelist) != 0
    applyPeriodWhitelist = len(plotPeriodWhitelist) != 0

    plotResultFiles = []
    for entry in dirList:
        filePath = os.path.join(plotResultDir, entry)
        entryArgs = entry[:-4].split('_')
        if os.path.isfile(filePath) and len(entryArgs) > 1 and entry.split('_')[0] == 'plotResult':
            numSwimmers = entryArgs[3].split('=')[1]
            rho = entryArgs[4].split('=')[1]
            amplitude = entryArgs[5].split('=')[1]
            period = entryArgs[6].split('=')[1]
            phaseShift = entryArgs[7].split('=')[1]

            if applyNumSwimmersWhitelist and numSwimmers not in plotNumSwimmersWhitelist:
                continue

            if applyRhoWhitelist and rho not in plotRhoWhitelist:
                continue

            if applyAmplitudeWhitelist and amplitude not in plotAmplitudeWhitelist:
                continue

            if applyPeriodWhitelist and period not in plotPeriodWhitelist:
                continue

            with open(filePath) as plotFile:
                resultObj = json.load(plotFile)
                x = resultObj[attributeAsX]
                y = resultObj['absoluteVelocity']
                std = resultObj['std']

            plotResultFiles.append(
                {'entry': entry, 'numSwimmers': numSwimmers, 'rho': rho, 'amplitude': amplitude, 'period': period,
                 'phaseShift': phaseShift, 'x': x, 'y': y, 'std': std})

    colors = cm.tab20(range(len(plotResultFiles)))

    for i, plotResultEntry in enumerate(plotResultFiles):
        numSwimmers = plotResultEntry['numSwimmers']
        rho = plotResultEntry['rho']
        amplitude = plotResultEntry['amplitude']
        period = plotResultEntry['period']
        phaseShift = plotResultEntry['phaseShift']
        if int(numSwimmers) == 400:
            marker='^'
        elif int(numSwimmers) == 1000:
            marker='x'
        else:
            marker='D'

        plt.errorbar(plotResultEntry['x'], plotResultEntry['y'], yerr=plotResultEntry['std'], fmt= matplotlib.colors.rgb2hex(colors[i]), marker=marker, linestyle='none',
                     label=f'$N={numSwimmers}, \\varrho={rho}, A={amplitude}, T={period}, \\Delta \\varphi={phaseShift}$')

    # plt.xlabel(r'$\rho \longrightarrow$')
    plt.xlabel(r'$\eta \longrightarrow$')
    # plt.xlabel(r'$A \longrightarrow$')
    plt.ylabel(r'$v_a \longrightarrow$')
    plt.grid('both')
    plt.legend()

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 5)

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
