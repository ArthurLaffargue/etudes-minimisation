import numpy as np
import sys
import matplotlib.pyplot as plt
plt.rc('font',family='Serif')

sys.path.append("../../minimisation")
from _genetic_algorithm import continousSingleObjectiveGA

from heatPumpSimulation import heatPump
## Optimisation
Taeo = 273.15 + 5.00
Te = 273.15 - 5.00
Tc = 273.15 + 40.00

xmin = np.array([0,-30,30])+273.15
xmax = np.array([20,-2.0,60.0])+273.15
xinit = [Taeo,Te,Tc]



sim = heatPump()
costFunction = lambda x : sim.cost(x)

cons = [{'type': 'ineq', 'fun': sim.contrainte1},
        {'type': 'ineq', 'fun': sim.contrainte2},
        {'type': 'ineq', 'fun': sim.contrainte3},
        {'type': 'ineq', 'fun': sim.contrainte4},
        {'type': 'ineq', 'fun': sim.contrainte5},
        {'type': 'ineq', 'fun': sim.contrainte6}]


npop = 50
ngen = 300
minAg = continousSingleObjectiveGA(costFunction,xmin,xmax,constraints=cons,preprocess_function=sim.simulateHeatPump,stagThreshold=100)

Xag,Yag = minAg.minimize(npop,ngen,verbose=False)
fitnessArray = minAg.getStatOptimisation()
fitnessArray = fitnessArray[fitnessArray != None]

sim.printDictSim(Xag)

print(Xag)
print(Yag)
print("Function calls : ",npop*len(fitnessArray))

plt.figure(figsize=(8,4))
plt.plot(fitnessArray,label='fmin',marker='o',ls='--',markeredgecolor='k',markerfacecolor="y",color='grey')
plt.grid(True)
plt.xlabel("Nombre de générations")
plt.ylabel("Fonction objectif")
plt.title("Convergence de la solution")
plt.legend(loc=0)
plt.tight_layout()
plt.savefig("convergence.svg",dpi=300)

plt.show()