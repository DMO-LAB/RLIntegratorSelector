from ember import Config, Paths, InitialCondition, StrainParameters, General, Times, TerminationCondition, ConcreteConfig, General, Debug, RK23Tolerances
import matplotlib.pyplot as plt

output = 'run/ex_diffusion9'
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cantera as ct
import os
import time
from ember import _ember



conf = Config(
    Paths(outputDir='run/ex_diffusion19'),
    General(nThreads=1,
            chemistryIntegrator='qss'),
    InitialCondition(Tfuel=600,
                     Toxidizer=1200,
                     centerWidth=0.0,
                     equilibrateCounterflow=False,
                     flameType='diffusion',
                     slopeWidth=0.0,
                     xLeft=-0.02,
                     pressure=101325,
                     xRight=0.02,
                     nPoints=100),
    StrainParameters(final=100,
                     initial=100),
    Times(globalTimestep=1e-05,
          profileStepInterval=20),
    TerminationCondition(abstol=0.0,
                         dTdtTol=0,
                         steadyPeriod=1.0,
                         tEnd=0.08,
                         tolerance=0.0),
    RK23Tolerances(
        absoluteTolerance=1e-8,
        relativeTolerance=1e-6,
        minimumTimestep=1e-10,
        maximumTimestep=1e-4,
        maxStepsNumber=100000,
    ),
    Debug(veryVerbose=False),)


conf = ConcreteConfig(conf)

confString = conf.original.stringify()

if not os.path.isdir(conf.paths.outputDir):
    os.makedirs(conf.paths.outputDir, 0o0755)
confOutPath = os.path.join(conf.paths.outputDir, 'config')
if (os.path.exists(confOutPath)):
    os.unlink(confOutPath)
confOut = open(confOutPath, 'w')
confOut.write(confString)


import resource
def print_memory_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Memory usage: {usage / 1024:.2f} MB")
    

def initialize_solver(solver):
    solver.set_integrator_types(['cvode'] * 100)
    solver.step()
    
def run_simulation(integrator_type=None):
    solver = _ember.FlameSolver(conf)
    solver.initialize()
    
    initialize_solver(solver)
    
    if integrator_type is None:
        integrator_type = np.random.choice(['boostRK'], size=100)  
        integrator_type = integrator_type.tolist()
    else:
        integrator_type = integrator_type
    solver.set_integrator_types(integrator_type)

    for i in range(10):
        #print(f"Iteration {i} ............................")
        start_time = time.time()
        done = solver.step()
        time_taken = time.time() - start_time
        print_memory_usage()
        #print(f"TIME TAKEN: {time_taken}") 
        
    #print(integrator_type)

integrator_type = ['cvode', 'cvode', 'boostRK', 'boostRK', 'cvode', 'boostRK', 'boostRK', 'boostRK', 'cvode', 'boostRK', 'cvode', 'cvode', 'cvode', 'cvode', 'cvode', 'cvode', 'cvode', 'boostRK', 'boostRK', 'boostRK', 'cvode', 'boostRK', 'cvode', 'cvode', 'boostRK', 'cvode', 'cvode', 'boostRK', 'cvode', 'cvode', 'cvode', 'boostRK', 'cvode', 'cvode', 'cvode', 'boostRK', 'boostRK', 'cvode', 'boostRK', 'boostRK', 'boostRK', 'cvode', 'cvode', 'cvode', 'boostRK', 'boostRK', 'boostRK', 'cvode', 'cvode', 'cvode', 'cvode', 'boostRK', 'cvode', 'cvode', 'cvode', 'cvode', 'cvode', 'cvode', 'boostRK', 'boostRK', 'boostRK', 'cvode', 'cvode', 'boostRK', 'boostRK', 'boostRK', 'boostRK', 'cvode', 'boostRK', 'boostRK', 'cvode', 'boostRK', 'boostRK', 'boostRK', 'boostRK', 'boostRK', 'cvode', 'cvode', 'boostRK', 'boostRK', 'boostRK', 'cvode', 'cvode', 'cvode', 'boostRK', 'cvode', 'cvode', 'cvode', 'cvode', 'boostRK', 'cvode', 'boostRK', 'boostRK', 'cvode', 'cvode', 'cvode', 'boostRK', 'boostRK', 'cvode', 'boostRK']


#run_simulation(integrator_type)
for i in range(100):
    print(f"Iteration {i} ............................")
    run_simulation(integrator_type=None)