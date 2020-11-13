import numpy as np
import requests
import time
import matplotlib.pyplot as plt
from threading import Thread

# lamda = kids
# mu = progenitores

# NETWORKING
URL_PATH = "http://163.117.164.219/age/robot4?"
N_THREADS = 10
MAX_RETRIES = 10

# CONSTANTES
DNA_SIZE = 4                             # Corresponds to the number of motors
DNA_BOUND = [-180, 180]                  # Upper and lower bounds for DNA values
POPULATION_SIZE = 100                     # Population size
N_GENERATIONS = 100
N_KID = 200                               # n kids per generation = lambda
tau = 1/np.sqrt(2*np.sqrt(DNA_SIZE))     # consideramos b = 1
tau0 = 1/np.sqrt(2*DNA_SIZE)             # consideramos b = 1
size_tournament = 5

# Plotting
PLOTTING_REAL_TIME = 1  # Choose to show fitness plot in real time
generations_plt = []    # Plotting axis
fitness_curve = []      # Plotting curve

def process_individual(session, url):
    """ process a single individual """
    return float(session.get(url).content)

def process_id_range(id_range, urls, session, store=None):
    """process a number of urls, storing the results in an array"""
    if store is None:
        store = {}
        print("jua jua jua")
    for id in id_range:
        store[id] = process_individual(session, urls[id])
    return store

def get_urls(population):
    all_urls = []
    for row, ind in enumerate(population['DNA']):
        url = URL_PATH
        for rotor in range(0,len(ind)):
            # This is for getting the fitness of an specific individual
            url = url + str("c")+str(rotor+1)+str("=")+str(ind[rotor])+str("&")
        url = url[:-1]

        all_urls.append(url)
    return all_urls

def evaluation(nthreads, population, session):
    """process the population in a specified number of threads"""
    id_range = range(len(population['DNA']))
    fitnesses = {}
    threads = []
    urls = get_urls(population)
    # create the threads
    for i in range(nthreads):
        ids = id_range[i::nthreads]
        t = Thread(target=process_id_range, args=(ids, urls, session, fitnesses))
        threads.append(t)
    
    # start the threads
    [ t.start() for t in threads ]
    # wait for the threads to finish
    [ t.join() for t in threads ]
    
    array = np.empty(len(population['DNA']))
    for i in range(len(population['DNA'])):
        array[i]=fitnesses[i]
    
    return array


# c1=3.412&c2=2.4&c3=15.42312&c4=-23.412235
# def good_evaluation(population, session):

#     fitness_population = np.empty(POPULATION_SIZE)


#     for row, ind in enumerate(population['DNA']):
#         url = URL_PATH
#         for rotor in range(0,len(ind)):
#             # This is for getting the fitness of an specific individual
#             url = url + str("c")+str(rotor+1)+str("=")+str(ind[rotor])+str("&")
#         url = url[:-1]

#         try:
#             r = session.get(url).content
#         except:
#             print("Exception when calling web service")
#             time.sleep(1)
#             r = session.get(url).content
#         fitness_population[row] = float(r)
#     return fitness_population

def tournament(population, size_tournament, num_parents):
    selected = []
    for _ in range(num_parents):
        winner = np.random.choice(np.arange(POPULATION_SIZE), size=size_tournament, replace=False)
        selected.append(np.max(winner))
    return selected
    

def make_kid(population, n_kid):
    # generate empty kid holder
    kids = {'DNA': np.empty((n_kid, DNA_SIZE))}
    kids['mut_strength'] = np.empty_like(kids['DNA'])
    
    for i in range(n_kid):
        ks = np.empty(DNA_SIZE) # initialize
        kv = np.empty(DNA_SIZE) # initialize
        
        parents = tournament(population['DNA'], size_tournament, 2)
        p1 = parents[0]
        p2 = parents[1]

        # uniform crossing with average among parents
        kv = (population['DNA'][p1] + population['DNA'][p2]) / 2
        # cruce posicional de varianzas
        cp = np.random.randint(0, 2, DNA_SIZE, dtype=np.bool)  # crossover points
        ks[cp] = population['mut_strength'][p1, cp]
        ks[~cp] = population['mut_strength'][p2, ~cp]

        # DNA mutation 
        kv = np.random.normal(kv,ks)
        # variances mutation 
        ks = np.dot(ks, np.dot(np.exp(np.random.normal(0,tau,DNA_SIZE)), np.exp(np.random.normal(0,tau0,DNA_SIZE))))
        
        # clip the mutated value
        kv[:] = np.clip(kv, *DNA_BOUND)

        # update kids array
        kids['DNA'][i] = kv
        kids['mut_strength'][i] = ks

    return kids

# def get_fitness(pred): return pred.flatten()

def kill_bad(pop, kids, session):
    # put population and kids together
    for key in ['DNA', 'mut_strength']:
        pop[key] = np.vstack((pop[key], kids[key]))

    fitness = evaluation(N_THREADS, pop, session)            # calculate global fitness
    idx = np.arange(pop['DNA'].shape[0])
    index_sort = fitness.argsort()
    good_idx = idx[index_sort][:POPULATION_SIZE]   # selected by fitness ranking (not value)
    for key in ['DNA', 'mut_strength']:
        pop[key] = pop[key][good_idx]
    return pop, fitness[index_sort]

def main():
    # initialize randomly the population DNA values (motor angles)
    # initialize randomly the variances with big values
    population = dict(DNA=np.random.uniform(low=DNA_BOUND[0], high=DNA_BOUND[1], size=(POPULATION_SIZE,DNA_SIZE) ),   
            mut_strength=np.random.rand(POPULATION_SIZE, DNA_SIZE)*DNA_BOUND[1])                                     

    
    # for plotting
    if PLOTTING_REAL_TIME == 1:
        plt.plot(generations_plt, fitness_curve, 'b', linewidth=1.0, label='Best individual fitness')
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.legend()

    # create Session with server
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=MAX_RETRIES)
    session.mount('https://', adapter)
    session.mount('http://', adapter)

    a=evaluation(N_THREADS, population, session)
    # b=good_evaluation(population, session)

    for i in range(N_GENERATIONS):
        # print(population['mut_strength'])

        kids = make_kid(population, N_KID)
        population, fitness_pop = kill_bad(population, kids, session)   # keep some good parent for elitism
        # fitness_population = evaluation(population)
        

        if PLOTTING_REAL_TIME == 1:
            generations_plt.append(i)
            fitness_curve.append(fitness_pop[0])
            plt.plot(generations_plt, fitness_curve, 'b', linewidth=1.0, label='Best individual fitness')
            plt.pause(0.05)


if __name__ == "__main__":
    main()