import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plotResultFileDir1 = 'D:/simulationdata/plotResultSameRhoGroupWOO400.txt'
    plotResultFileDir2 = 'D:/simulationdata/plotResultSameRho1000_2.txt'
    # plotResultFileDir3 = 'D:/simulationdata/plotResultsameRhoSameEtaGroup4000.txt'

    with open(plotResultFileDir1) as plotFile:
        resultObj400 = json.load(plotFile)

    with open(plotResultFileDir2) as plotFile:
        resultObj1000 = json.load(plotFile)

    # with open(plotResultFileDir3) as plotFile:
    #     resultObj4000 = json.load(plotFile)

    plt.plot(resultObj400['x'], resultObj400['y'], 'ro', label='N = 400')
    plt.plot(resultObj1000['x'], resultObj1000['y'], 'gx', label='N = 1000')
    # plt.plot(resultObj4000['x'], resultObj4000['y'], 'b*', label='N = 4000')
    # plt.xlabel(r'$\rho \longrightarrow$')
    plt.xlabel(r'$\eta \longrightarrow$')
    # plt.xlabel(r'$A \longrightarrow$')
    plt.ylabel(r'$v_a \longrightarrow$')
    plt.grid('both')
    plt.legend()
    plt.show()