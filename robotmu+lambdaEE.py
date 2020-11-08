import numpy as np
import requests
import time
import matplotlib.pyplot as plt

#TODO: Nos hemos quedado en la implementación correcta del cruze!!! 
# Falta todo lo otro

# Apuntes:
# lamda = kids
# mu = progenitores


# CONSTANTES
DNA_SIZE = 4                  # Corresponds to the number of motors
DNA_BOUND = [-180, 180]       # Upper and lower bounds for DNA values
POPULATION_SIZE = 10          # Population size
N_GENERATIONS = 100
N_KID = 50                    # n kids per generation = lambda
tau = 1/np.sqrt(DNA_SIZE)


# c1=3.412&c2=2.4&c3=15.42312&c4=-23.412235
def evaluation(population):
    url_original = "http://163.117.164.219/age/robot4?"
    fitness_population = np.empty([len(population['DNA']),DNA_SIZE])
    for row, ind in enumerate(population['DNA']):
        url = url_original
        for rotor in range(0,len(ind)):
            # This is for getting the fitness of an specific individual
            url = url + str("c")+str(rotor+1)+str("=")+str(ind[rotor])+str("&")
        url = url[:-1]
        for rotor in range(0,len(ind)):
            try:
                r = requests.get(url).content
            except:
                print("Exception when calling web service")
                time.sleep(1)
                r = requests.get(url).content
            fitness_population[row][rotor] = float(r)
    return fitness_population

def make_kid(population, n_kid):
    # generate empty kid holder
    kids = {'DNA': np.empty((n_kid, DNA_SIZE))}
    kids['mut_strength'] = np.empty_like(kids['DNA'])
    
    for kv, ks in zip(kids['DNA'], kids['mut_strength']):
        # crossover (roughly half p1 and half p2)
        # randomly choose 2 parents (indexes)
        p1, p2 = np.random.choice(np.arange(POPULATION_SIZE), size=2, replace=False)
        print(p1)
        print(p2)
        print(kv)
        print(ks)

        # cp = np.random.randint(0, 2, DNA_SIZE, dtype=np.bool)  # crossover points
        # kv[cp] = population['DNA'][p1, cp]
        # kv[~cp] = population['DNA'][p2, ~cp]
        # ks[cp] = population['mut_strength'][p1, cp]
        # ks[~cp] = population['mut_strength'][p2, ~cp]

        # uniform crossing with average among parents
        kv = (population['DNA'][p1] + population['DNA'][p2]) / 2
        ks = np.sqrt(population['mut_strength'][p1] + population['mut_strength'][p2])

        # DNA mutation 
        kv = np.random.normal(kv,ks)
        # variances mutation 
        ks = np.exp(np.multiply(np.random.normal(0,tau),ks))

        # clip the mutated value    
        kv[:] = np.clip(kv, *DNA_BOUND)    

        # mutate (change DNA based on normal distribution)
        # ks[:] = np.maximum(ks + (np.random.rand(*ks.shape)-0.5), 0.)    # must > 0
        # kv += ks * np.random.randn(*kv.shape)

    return kids

def get_fitness(pred): return pred.flatten()

def kill_bad(pop, kids):
    # put population and kids together
    for key in ['DNA', 'mut_strength']:
        pop[key] = np.vstack((pop[key], kids[key]))

    fitness = get_fitness(evaluation(population))            # calculate global fitness
    idx = np.arange(pop['DNA'].shape[0])
    good_idx = idx[fitness.argsort()][-POP_SIZE:]   # selected by fitness ranking (not value)
    for key in ['DNA', 'mut_strength']:
        pop[key] = pop[key][good_idx]
    return pop



# initialize randomly the population DNA values (motor angles)
# initialize randomly the variances with big values
population = dict(DNA=np.random.uniform(low=DNA_BOUND[0], high=DNA_BOUND[1], size=(POPULATION_SIZE,DNA_SIZE) ),   
           mut_strength=np.random.rand(POPULATION_SIZE, DNA_SIZE)*DNA_BOUND[1])                                     

print(population)

for _ in range(N_GENERATIONS):

    fitness_population = evaluation(population)
    # print(fitness_population)


    kids = make_kid(population, N_KID)

    print(kids)
    population = kill_bad(population, kids)   # keep some good parent for elitism





# Teacher comments:
# P: en σ′=eN(0,τ)·σ, cuanto vale τ ?

# R: es un parámetro que se llama tasa de aprendizaje
#    debes jugar con el un poco
#    debería de ser aprox 1/sqrt(n)
#    siendo n la longitud del cromosoma