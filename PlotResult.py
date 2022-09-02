import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plotResultFileDir1 = 'E:/simulationdata/phaseShift/samerho400/plotResultNoPhaseShift.txt'
    plotResultFileDir2 = 'E:/simulationdata/phaseShift/samerho400/plotResult90Phase.txt'
    plotResultFileDir3 = 'E:/simulationdata/phaseShift/samerho400/plotResult180Phase.txt'

    with open(plotResultFileDir1) as plotFile:
        resultObj400 = json.load(plotFile)

    with open(plotResultFileDir2) as plotFile:
        resultObj400_90 = json.load(plotFile)

    with open(plotResultFileDir3) as plotFile:
        resultObj400_180 = json.load(plotFile)

    plt.errorbar(resultObj400['x'], resultObj400['y'], yerr=resultObj400['std'], fmt='ro', label='$\Delta \phi = 0$')
    plt.errorbar(resultObj400_90['x'], resultObj400_90['y'], yerr=resultObj400_90['std'], fmt='gx', label='$\Delta \phi = 90$')
    plt.errorbar(resultObj400_180['x'], resultObj400_180['y'], yerr=resultObj400_180['std'], fmt='b*', label='$\Delta \phi = 180$')
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