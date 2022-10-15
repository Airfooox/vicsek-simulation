import os
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors

if __name__ == "__main__":
    plotResultDir = "C:\\Users\\konst\\OneDrive\\Uni\\Anstellung\\Prof. Menzel (2020-22)\\vicsek\\daten\\parameter_test\\N=400"
    dirList = os.listdir(plotResultDir)

    attributeAsX = 'randomAngleAmplitude'

    plotResultFiles = []
    for entry in dirList:
        filePath = os.path.join(plotResultDir, entry)
        entryArgs = entry[:-4].split('_')
        if os.path.isfile(filePath) and len(entryArgs) > 1 and entry.split('_')[0] == 'plotResult':
            numSwimmers = entryArgs[2].split('=')[1]
            rho = entryArgs[3].split('=')[1]
            amplitude = entryArgs[4].split('=')[1]
            period = entryArgs[5].split('=')[1]

            with open(filePath) as plotFile:
                resultObj = json.load(plotFile)
                x = resultObj[attributeAsX]
                y = resultObj['absoluteVelocity']
                std = resultObj['std']

            plotResultFiles.append(
                {'entry': entry, 'numSwimmers': numSwimmers, 'rho': rho, 'amplitude': amplitude, 'period': period,
                 'x': x, 'y': y, 'std': std})

    colors = cm.tab20(range(20))

    for i, plotResultEntry in enumerate(plotResultFiles):
        numSwimmers = plotResultEntry['numSwimmers']
        rho = plotResultEntry['rho']
        amplitude = plotResultEntry['amplitude']
        period = plotResultEntry['period']
        if int(numSwimmers) == 400:
            marker='^'
        elif int(numSwimmers) == 1000:
            marker='x'
        else:
            marker='D'

        plt.errorbar(plotResultEntry['x'], plotResultEntry['y'], yerr=plotResultEntry['std'], fmt= matplotlib.colors.rgb2hex(colors[i]), marker=marker, linestyle='none',
                     label=f'$N={numSwimmers}, \\varrho={rho}, A={amplitude}, T={period}$')

    # plt.xlabel(r'$\rho \longrightarrow$')
    plt.xlabel(r'$\eta \longrightarrow$')
    # plt.xlabel(r'$A \longrightarrow$')
    plt.ylabel(r'$v_a \longrightarrow$')
    plt.grid('both')
    plt.legend()
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
