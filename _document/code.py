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

def eval_fitness(fobj):
    fmin,fmax = fobj.min(),fobj.max()
    if fmin == fmax : 
        fitness = np.ones_like(fobj,dtype=float)
    else : 
        fitness = (fmax-fobj)/(fmax-fmin)
    return fitness

def tournament_selection(fitness,population):
    npop,ndof = population.shape
    selection = np.zeros((npop//2,ndof),dtype=float)
    for i in range(npop//2):
        indices = rd.choice(npop-1,2)
        best = indices[np.argmax(fitness[indices])]
        selection[i] = population[best]
    return selection

def normalCrossover(selection,delta_cross):
    nsel,ndof = selection.shape
    npop = nsel*2
    couples = np.zeros((nsel,2,ndof))
    children = np.zeros((npop,ndof))
    across = rd.normal(size=(nsel,ndof))*delta_cross
    for i in range(nsel):
        k = i
        while k == i :
            k = rd.randint(0,nsel-1)
        couples[i] = [selection[i],selection[k]]
    children[:npop//2] = across*couples[:,0] + (1-across)*couples[:,1]
    children[npop//2:] = across*couples[:,1] + (1-across)*couples[:,0]

    return children

def normalMutation(population,delta_mut,rho_mut) :
    npop,ndof = population.shape
    probaMutation = rd.sample((npop,ndof))
    deltaX = delta_mut*( rd.normal(size=(npop,ndof)) )
    population = population + deltaX*(probaMutation<=rho_mut)

    return population


def elitisme(population_xs,population_f,elite_population,elite_objective) : 
    if elite_objective is None : 
        elite_objective = population_f
        elite_population = population_xs
    else : 
        npop = population_xs.shape[0]
        union_pop = np.concatenate([elite_population,population_xs],axis=0)
        union_obj = np.concatenate([elite_objective,population_f])

        sortindex = np.argsort(union_obj)

        population_xs = union_pop[sortindex][:npop]
        population_f = union_obj[sortindex][:npop]

        elite_objective = union_obj[sortindex]
        elite_population = union_pop[sortindex]

        if elite_population.shape[0] > (2*npop) : 
            elite_objective = elite_objective[:2*npop]
            elite_population = elite_population[:2*npop]
    
    return population_xs,population_f,elite_population,elite_objective


def minimize_genetic_algorithm(func,xmin,xmax,popsize=50,maxiter=200):
    delta_mut = 0.10
    delta_cross = 1.20
    rho_mut = 0.50
    tol = 0.0001
    atol = 0.0

    if popsize%2 != 0 :
        popsize += 1

    ndof = len(xmin)
    population_xs = latin_hypercube_sampling((popsize,ndof)) 
    population_x = population_xs* (xmax-xmin) + xmin
    population_f = np.array([func(xi) for xi in population_x])

    bestarg = np.argmin(population_f)
    best_x = population_x[bestarg]
    best_f = population_f[bestarg]

    elite_population = None 
    elite_objective = None

    for iter in range(maxiter): 
        
        fitness = eval_fitness(population_f)
        selection = tournament_selection(fitness,population_xs)
        population_xs[:] = normalCrossover(selection,delta_cross)
        population_xs = np.minimum(np.maximum(0.0,population_xs),1.0)
        population_xs[:] = normalMutation(population_xs,delta_mut,rho_mut)
        population_xs = np.minimum(np.maximum(0.0,population_xs),1.0)


        population_x = population_xs* (xmax-xmin) + xmin
        population_f = np.array([func(xi) for xi in population_x])

        bestarg = np.argmin(population_f)
        best_x_iter = population_x[bestarg]
        best_f_iter = population_f[bestarg]

        if best_f_iter < best_f : 
            best_f = best_f_iter
            best_x[:] = best_x_iter[:]

        (population_xs,
        population_f,
        elite_population,
        elite_objective) = elitisme(population_xs,
                                    population_f,
                                    elite_population,
                                    elite_objective)


        #convergence 
        if np.std(population_f) <= (atol + tol*np.abs(np.mean(population_f))) : 
            print("SOLUTION CONVERGED : iter %i"%iter)
            break 

    return best_x



if __name__ == '__main__' : 
    func = lambda x : (-(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1]  + 47))))
                -x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]  + 47)))))

    
    xmin = [-75,-75]
    xmax = [75,75]
    lw = np.array(xmin)
    up = np.array(xmax)

    xopt = minimize_genetic_algorithm(func,lw,up,popsize=40,maxiter=50)
    fopt = func(xopt)

    #[-46.45093036  38.12486808] 
    # -126.42383220971709

    print(xopt)
    print(fopt)





            




