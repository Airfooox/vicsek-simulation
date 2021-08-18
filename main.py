# setup
from classes import Simulation
from util import printProgressBar

if __name__ == "__main__":
    CONSTANTS = {
        "fps": 30,
        "time": 30,
        "environmentSideLength": 10,
        "numSwimmers": 300,
        "initialVelocity": 0.3,

        "swimmerSize": 0.04,
        "saveVideo": False
    }

    simulation = Simulation(CONSTANTS)
    printProgressBar(0, simulation.numFrames, prefix='Simulation Progress:', suffix='Simulation Complete', length=50)
    simulation.simulate()
    simulation.animate()
