import numpy as np
import numpy.random as rd


def autoSetUp(func,ndof,maxiter):
    # Routine de
    # 100 permutations aléatoires
    rd_xarray = np.array([rd.sample(ndof)*(xmax-xmin) + xmin for i in range(100)])
    farray = np.array([func(xi) for xi in rd_xarray])
    # Evaluation de la différence d'energie
    deltaE = farray[1:]-farray[:-1]
    # Valeur moyenne des valeurs positives
    deltaE_avg = np.mean(deltaE[deltaE>0])

    initialTemp = -deltaE_avg/np.log(0.60)  #Démarrage haute température
    finalTemp = -deltaE_avg/np.log(0.01)
    decreaseRatio = (initialTemp/finalTemp)**(1/maxiter)
    return initialTemp,decreaseRatio

def recuit_simule(func,xmin,xmax,maxiter=1000):

    # parametres
    perturbRatio = 0.5 #Ratio de déplacement maximale

    # solution initiale
    ndof = len(xmin)
    s0 = rd.sample(ndof)
    x0 = s0*(xmax-xmin) + xmin
    E0 = func(x0)

    # initialisation de la température et du taux de décroissance
    initialTemp,decreaseRatio = autoSetUp(func,ndof,maxiter)
    T = initialTemp

    # solution optimale
    xopt = x0[:]
    Eopt = E0

    for iter in range(maxiter):

        # perturbation de l'etat du systeme
        x1 = x0 + perturbRatio*2*(rd.sample(ndof)-0.5)
        x1 = np.minimum(xmax,np.maximum(xmin,x1))
        E1 = func(x1)
        deltaE = E1-E0

        # acceptation de la solution
        if deltaE < 0.0 :
            x0 = x1[:]
            E0 = E1

            # solution optimale
            if E0 < Eopt :
                xopt = x0[:]
                Eopt = E0

        # acceptation d'une solution dégradée
        elif rd.random() < np.exp(-deltaE/T) :
            x0 = x1[:]
            E0 = E1

        # réduction de la température
        T = T*decreaseRatio
    return xopt






