import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

from networkSimulation import hydroNetwork

plt.rc('font',family='Serif')


sys.path.append("../../minimisation")
from _minimize_gradient import *
from _minimize_BFGS import *

hydroSim = hydroNetwork()
minEconomicFactor = hydroSim.minEconomicFactor
maxEconomicFactor = hydroSim.maxEconomicFactor
Dmin = hydroSim.Dmin
Dmax = hydroSim.Dmax

niteration = 150
economicConstraint = np.linspace(0.0,
                                 1.0,
                                 niteration)

energyConstraint = np.linspace(0.0,1.0,niteration//2)

DijVector = []
energyVector = []
economicVector = []
constraintViolation = []
iterationVector = []
funcCallsVector = []
Dk = Dmin

# figNetwork = plt.figure()

t1 = time.time()
penalityFactor = 10000
for k,ecoCons in enumerate(economicConstraint) :

    # figNetwork.clf()

    print("iteration ",k+1,"/",niteration)
    cons = [{"type":"ineq",
             "fun":lambda Dij : ecoCons - hydroSim.economicCostFunc(Dij),
             "jac":lambda Dij : -hydroSim.economicCostGrad(Dij) }]

    dictMin = BFGS(hydroSim.energyCostFunc,
                        Dk,
                        Dmin,
                        Dmax,
                        gf = hydroSim.energyCostGrad,
                        returnDict=True,
                        constraints=cons,
                        penalityFactor=penalityFactor,
                        penalInnerIter = 1,
                        tol=1e-4,
                        gtol=1e-4,
                        maxIter=250)

    Dk = dictMin["x"]

    phi_eco = hydroSim.economicCostFunc(Dk)
    phi_hydro = dictMin["fmin"]

    DijVector.append(Dk)
    energyVector.append(phi_hydro)
    economicVector.append(phi_eco)
    normViol = np.linalg.norm(dictMin["constrViolation"])
    constraintViolation.append(normViol)
    iterationVector.append(dictMin["iterations"])
    funcCallsVector.append(dictMin["functionCalls"])

    # for s in dictMin:
    #     if not( s.endswith("History") ):
    #         print(s," : ",dictMin[s])
    # print("#"*50)
    # print("\n")

    penalityFactor = max(100,0.9*penalityFactor)

    # title = r'Diamètres $\phi_{hydro} =$ %.3f and $\phi_{eco} = %.3f'%(phi_hydro,phi_eco)
    # hydroSim.plotUfieldBranch(Dk,
    #                         fig=figNetwork,
    #                         title=title,
    #                         nodeLabel=False,
    #                         cbarLabel='Diamètre [m]')
    # figNetwork.savefig("plotsnetwork/fig_no"+str(k)+".png")

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
method = dictMin["method"]
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
