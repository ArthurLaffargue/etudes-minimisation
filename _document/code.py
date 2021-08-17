import numpy as np
import numpy.random as rd

def latin_hypercube_sampling(shape):
    popsize = shape[0]
    ndof = shape[1]
    dsize  = 1.0/popsize

    sample_array = dsize*rd.sample(size=shape) 
    sample_array[:] += np.linspace(0.,1.,popsize)[:,np.newaxis]
    init_pop = np.zeros(shape)
    for dim in range(ndof) : 
        rdm_order = rd.permutation(popsize)
        init_pop[:,dim] = sample_array[rdm_order,dim]

    return init_pop

def recombinaison(index,population,scale_factor):
    #best1
    index_list = list(range(len(population)))
    index_list.remove(index)
    rd.shuffle(index_list)
    r1,r2 = index_list[:2]
    y = population[0,:] + \
        scale_factor*(population[r1,:]-population[r2,:])
    return y

def differential_evolution(func,xmin,xmax,popsize=20,maxiter=1000):
    atol,tol = 0.0,0.001
    CR = 0.9
    F = [0.5,2.0]

    ndof = len(xmin)
    population_x = latin_hypercube_sampling((popsize,ndof)) * (xmax-xmin) + xmin
    population_f = np.array([func(xi) for xi in population_x])

    bestarg = np.argmin(population_f)
    population_x[[0,bestarg],:] = population_x[[bestarg,0],:] 
    population_f[[0,bestarg]] = population_f[[bestarg,0]] 
    best_f = population_f[0]

    for iter in range(maxiter): 
        
        scale_factor = rd.sample(ndof)*(F[1]-F[0]) + F[0]
        for k,xk in enumerate(population_x) :
            #recombinaison
            x_prime = recombinaison(k,population_x,scale_factor)
            x_prime = np.maximum(xmin,np.minimum(x_prime,xmax))

            #croisement
            croisement = rd.sample(size=ndof) <= CR 
            x_trial = np.where(croisement,x_prime,xk)

            #selection
            fk = population_f[k]
            f_trial = func(x_trial)
            if f_trial < fk : 
                population_x[k] = x_trial
                population_f[k] = f_trial

                if f_trial < best_f : 
                    population_x[[0,k],:] = population_x[[k,0],:] 
                    population_f[[0,k]] = population_f[[k,0]] 
                    best_f = population_f[0]
        
        #convergence 
        if np.std(population_f) <= (atol + tol*np.abs(np.mean(population_f))) : 
            print("SOLUTION CONVERGED : iter %i"%iter)
            break 

    return population_x[0]



if __name__ == '__main__' : 
    func = lambda x: np.sum(x*x - 10*np.cos(2*np.pi*x)) + 10*np.size(x)
    lw = np.array([-5.12] * 5)
    up = np.array([5.12] * 5)

    xopt = differential_evolution(func,lw,up,popsize=50,maxiter=1000)
    fopt = func(xopt)

    print(xopt)
    print(fopt)





            




