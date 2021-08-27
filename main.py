# setup
from classes import Simulation
from util import printProgressBar

if __name__ == "__main__":
    CONSTANTS = {
        "fps": 60,
        "time": 30,
        "environmentSideLength": 15,
        "numSwimmers": 1000,
        "initialVelocity": 0.3,

        "swimmerSize": 0.04,
        "saveVideo": False
    }

    simulation = Simulation(CONSTANTS)
    printProgressBar(0, simulation.numFrames, prefix='Simulation Progress:', suffix='Simulation Complete', length=50)
    simulation.simulate()
    simulation.animate()
