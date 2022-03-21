import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plotResultFileDir1 = 'E:/simulationdata/bla/plotResultSameEta.txt'
    # plotResultFileDir2 = 'E:/simulationdata/bla/plotResultSameRho1000NoOci.txt'
    # plotResultFileDir3 = 'E:/simulationdata/bla/plotResultSameRho1000Oci.txt'

    with open(plotResultFileDir1) as plotFile:
        resultObj400 = json.load(plotFile)

    # with open(plotResultFileDir2) as plotFile:
    #     resultObj1000 = json.load(plotFile)
    #
    # with open(plotResultFileDir3) as plotFile:
    #     resultObj4000 = json.load(plotFile)

    plt.errorbar(resultObj400['x'], resultObj400['y'], yerr=resultObj400['std'], fmt='ro', label='$\eta$ = 2')
    # plt.errorbar(resultObj1000['x'], resultObj1000['y'], yerr=resultObj1000['std'], fmt='gx', label='N = 1000')
    # plt.errorbar(resultObj4000['x'], resultObj4000['y'], yerr=resultObj4000['std'], fmt='b*', label='N = 1000, Osci')
    plt.xlabel(r'$\rho \longrightarrow$')
    # plt.xlabel(r'$\eta \longrightarrow$')
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