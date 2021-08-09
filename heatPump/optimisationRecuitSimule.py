import numpy as np
from scipy.optimize import minimize
import sys
import matplotlib.pyplot as plt
plt.rc('font',family='Serif')

sys.path.append("../../minimisation")
from _simulated_annealing import minimize_simulatedAnnealing


from heatPumpSimulation import heatPump
## Optimisation
Taeo = 273.15 + 5.00
Te = 273.15 - 5.00
Tc = 273.15 + 40.00

xmin = np.array([0,-30,30])+273.15
xmax = np.array([20,-2.0,60.0])+273.15
xinit = [Taeo,Te,Tc]



sim = heatPump()

cons = [{'type': 'ineq', 'fun': sim.contrainte1},
        {'type': 'ineq', 'fun': sim.contrainte2},
        {'type': 'ineq', 'fun': sim.contrainte3},
        {'type': 'ineq', 'fun': sim.contrainte4},
        {'type': 'ineq', 'fun': sim.contrainte5},
        {'type': 'ineq', 'fun': sim.contrainte6}]


maxIter = 10000

mindict = minimize_simulatedAnnealing(sim.cost,
                                      xmin,
                                      xmax,
                                      maxIter=maxIter,
                                      constraints=cons,
                                      preprocess_function=sim.simulateHeatPump,
                                      autoSetUpIter=100,
                                      config="lowTemp",
                                      verbose=False,
                                      returnDict=True,
                                      storeIterValues=True)

Xsa = mindict['x']
fhistory = mindict["fHistory"]

sim.printDictSim(Xsa)

print(Xsa)
print(sim.cost(Xsa))

plt.figure(figsize=(8,4))
plt.plot(fhistory,label='fmin',marker='o',ls='--',markeredgecolor='k',markerfacecolor="r",color='grey')
plt.grid(True)
plt.xlabel("Nombre d'itérations")
plt.ylabel("Fonction objectif")
plt.title("Convergence de la solution")
plt.legend(loc=0)
plt.tight_layout()
plt.savefig("convergence.svg",dpi=300)

plt.show()