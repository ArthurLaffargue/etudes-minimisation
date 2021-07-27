import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

from networkSimulation import hydroNetwork

plt.rc('font',family='Serif')


hydroSim = hydroNetwork()
minEconomicFactor = hydroSim.minEconomicFactor
maxEconomicFactor = hydroSim.maxEconomicFactor
Dmin = hydroSim.Dmin
Dmax = hydroSim.Dmax

niteration = 150
economicConstraint = np.linspace(0.0,
                                 1.0,
                                 niteration)

DijVector = []
energyVector = []
economicVector = []
constraintViolation = []
iterationVector = []
funcCallsVector = []
Dk = Dmin
Dbounds = np.array([Dmin,Dmax]).T
options = {"maxiter":2000}
# figNetwork = plt.figure()
method = "trust-constr"

nfev = 0
def func(x):
    global nfev
    nfev += 1
    return hydroSim.energyCostFunc(x)


t1 = time.time()
for k,ecoCons in enumerate(economicConstraint) :

    # figNetwork.clf()

    print("iteration ",k+1,"/",niteration)
    cons = [{"type":"ineq",
             "fun":lambda Dij : ecoCons - hydroSim.economicCostFunc(Dij),
             "jac":lambda Dij : -hydroSim.economicCostGrad(Dij) }]

    resMin = minimize(func,
                    Dk,
                    jac = hydroSim.energyCostGrad,
                    bounds = Dbounds,
                    constraints=cons,
                    options = options,
                    method=method)

    Dk = resMin.x

    phi_eco = hydroSim.economicCostFunc(Dk)
    phi_hydro = resMin.fun

    DijVector.append(Dk)
    energyVector.append(phi_hydro)
    economicVector.append(phi_eco)
    normViol = 0.0
    constraintViolation.append(normViol)
    iterationVector.append(resMin.nit)
    funcCallsVector.append(resMin.nfev)

    print(resMin)
    print("#"*50)
    print("nfev = ",resMin.nfev)
    print("#"*50)
    print("\n")

t2 = time.time()

plt.figure("Pareto front")
plt.plot(economicVector,energyVector,'o')
plt.ylabel("econommicVector")
plt.ylabel("energyVector")
plt.grid(True)


plt.show()


resultOptimization = np.array([energyVector,
                               economicVector,
                               constraintViolation,
                               iterationVector,
                               funcCallsVector]).T
method = "scipy " + method
header = "energyFactor\t"
header += "econcomicFactor\t"
header += "constraintViolation\t"
header += "iteration\t"
header += "functionCalls"

np.savetxt("paretoFront/"+method+".txt",
            resultOptimization,
            header=header)


elapsedTime = t2-t1
print("elapsed time : ", elapsedTime)