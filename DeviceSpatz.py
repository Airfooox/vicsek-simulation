#setupimport randomimport numpy as npimport timeimport itertoolsfrom SimulationManager import SimulationManager, SimulationGroupif __name__ == "__main__":    starttime = time.perf_counter()    simulationDir = '/local/kzisiadis/vicsek-simulation_sameRho_phaseShift'    # simulationDir = 'D:/simulationdata'    simulationManager = SimulationManager(simulationDir)    simulationConfigs = [        {'numSwimmers': [400, 1000], 'rhos': [4], 'etas': [2], 'amplitudes': [np.pi / 64], 'periods': [50, 60, 70, 80, 90, 100],         'phaseShifts': [[0, np.pi / 2], [0, np.pi]]},    ]    for configEntry in simulationConfigs:        for config in itertools.product(configEntry['numSwimmer'], configEntry['rho'], configEntry['eta'],                                        configEntry['amplitude'], configEntry['period'], configEntry['phaseShifts']):            numSwimmer = config[0]            rho = config[1]            eta = config[2]            amplitude = config[3]            period = config[4]            phaseShifts = config[5]            environmentSideLength = np.sqrt(numSwimmer / rho)            def constantFunc(simulationIndex, numSimulation, n=numSwimmer, envL=environmentSideLength, e=eta,                             a=amplitude, per=period, pha=phaseShifts):                constants = {                    "timeSteps": 7200,                    "environmentSideLength": envL,                    "interactionRadius": 1,                    "randomAngleAmplitude": eta * (simulationIndex / numSimulation),                    "groups": {},                    "initialVelocity": 0.0025,                    "swimmerSize": 0.04,                    "saveVideo": False,                }                numGroups = len(phaseShifts)                for index, phaseShift in enumerate(phaseShifts):                    constants['groups'][f'{index}'] = {                        "numSwimmers": int(n // numGroups),                        "oscillationAmplitude": a,                        "oscillationPeriod": per,  # how many timesteps for one full oscillation                        "oscillationPhaseshift": phaseShift                    }                return constants            phaseShiftString = ','.join(f'{str(np.round(x / np.pi, 2))}pi' for x in phaseShifts)            simulationDataDir = simulationDir + f'/sameRhoGroup_phaseShift_numSwimmers={numSwimmer}_eta={eta}_amplitude={np.round(amplitude / np.pi, 2)}pi_period={period}_phaseShift=[{phaseShiftString}]'            sameRhoSimulationGroup = SimulationGroup(simulationDataDir=simulationDataDir,                                                     constantsFunc=constantFunc,                                                     numSimulation=30, repeatNum=25,                                                     timePercentageUsedForMean=25,                                                     saveTrajectoryData=False)            simulationManager.appendGroup(sameRhoSimulationGroup)    simulationManager.simulate()    print('That took {} seconds'.format(time.perf_counter() - starttime))