# setupimport randomimport osfrom Simulation import Simulationfrom util import printProgressBarimport numpy as npimport timeif __name__ == "__main__":    starttime = time.perf_counter()    simulationConfig = {        "timeSteps": 10000,        "environmentSideLength": np.sqrt(400 / 4),        "groups": {            "1": {                "numSwimmers": 200,                "oscillationAmplitude": np.pi / 16,                "oscillationPeriod": 40,  # how many timesteps for one full oscillation                "oscillationPhaseshift": 0            },            "2": {                "numSwimmers": 200,                "oscillationAmplitude": np.pi / 16,                "oscillationPeriod": 40,  # how many timesteps for one full oscillation                "oscillationPhaseshift": np.pi / 2            }        },        "interactionRadius": 1,        "randomAngleAmplitude": 0.1,        "velocity": 0.0025,        "swimmerSize": 0.04,        "saveVideo": False,    }    simulation = Simulation(simulationIndex=1, numSimulation=1, simulationConfig=simulationConfig, timePercentageUsedForMean=25)    print('Setup took {} seconds'.format(time.perf_counter() - starttime))    starttime = time.perf_counter()    simulation.simulate()    print(simulation.totalAbsoluteVelocity)    print(simulation.totalAbsoluteGroupVelocities)    print('Simulation took {} seconds'.format(time.perf_counter() - starttime))    # starttime = time.perf_counter()    # print('Absolute velocity took {} seconds'.format(time.perf_counter() - starttime))    # simulation.animate()