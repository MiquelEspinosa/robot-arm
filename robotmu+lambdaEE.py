import numpy as np
import requests
import time
import matplotlib.pyplot as plt
import concurrent.futures as futures
from threading import Thread

# lamda = kids
# mu = progenitores

# NETWORKING
URL_PATH = "http://163.117.164.219/age/robot10?"
# N_THREADS = 10
MAX_RETRIES = 10

# CONSTANTES
DNA_SIZE = 10                             # Corresponds to the number of motors
DNA_BOUND = [-180, 180]                  # Upper and lower bounds for DNA values
POPULATION_SIZE = 100                     # Population size
N_GENERATIONS = 1000
N_KID = 200                               # n kids per generation = lambda
tau = 1/np.sqrt(2*np.sqrt(DNA_SIZE))     # consideramos b = 1
tau0 = 1/np.sqrt(2*DNA_SIZE)             # consideramos b = 1
size_tournament = 5

# adaptation one-fifth rule
previous_fitness_pop = np.ones(POPULATION_SIZE)
c = 0.82                                 # constante para la regla de 1/5
# s = 15                                   # tamaño ventana para array de mejoras
# success = np.zeros(s)                 # guardamos el número de mejoras por cada s iteraciones

# Plotting
PLOTTING_REAL_TIME = 1  # Choose to show fitness plot in real time
generations_plt = []    # Plotting axis
fitness_curve = []      # Plotting curve
fitness_curve2 = []      # Plotting curve
fitness_curve3 = []      # Plotting curve
save_results = 'output'

def process_individual(session, individual):
    url = URL_PATH
    for rotor in range(0,len(individual)):
        url = url + str("c")+str(rotor+1)+str("=")+str(individual[rotor])+str("&")
    url = url[:-1]
    try:
        r = session.get(url).content
    except:
        print("Exception when calling web service")
        time.sleep(1)
        r = session.get(url).content
    return float(r)

def evaluation(population, session):
    pop_size = len(population['DNA'])
    population_fitness = np.empty(pop_size)
    with futures.ThreadPoolExecutor(max_workers=POPULATION_SIZE) as executor:
        future = [
            executor.submit(process_individual, session, ind)
            for ind in population['DNA']
        ]
    population_fitness = [f.result() for f in future]
    return np.array(population_fitness)

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
        ks = np.empty(DNA_SIZE) # initialize array
        kv = np.empty(DNA_SIZE) # initialize array        
        parents = tournament(population['DNA'], size_tournament, 2)
        p1 = parents[0]
        p2 = parents[1]

        # uniform crossing with average among parents
        kv = (population['DNA'][p1] + population['DNA'][p2]) / 2
        # cruce posicional de varianzas
        cp = np.random.randint(0, 2, DNA_SIZE, dtype=np.bool)  # crossover points
        ks[cp] = population['mut_strength'][p1, cp]
        ks[~cp] = population['mut_strength'][p2, ~cp]

        # DNA and variances mutation 
        kv = np.random.normal(kv,ks)
        ks = np.dot(ks, np.dot(np.exp(np.random.normal(0,tau,DNA_SIZE)), np.exp(np.random.normal(0,tau0,DNA_SIZE))))
        
        # clip the mutated value
        kv[:] = np.clip(kv, *DNA_BOUND)

        # update kids array
        kids['DNA'][i] = kv
        kids['mut_strength'][i] = ks

    return kids


def kill_bad(pop, kids, session):
    # put population and kids together
    for key in ['DNA', 'mut_strength']:
        pop[key] = np.vstack((pop[key], kids[key]))

    fitness = evaluation(pop, session)            # calculate global fitness
    idx = np.arange(pop['DNA'].shape[0])
    index_sort = fitness.argsort()
    good_idx = idx[index_sort][:POPULATION_SIZE]   # selected by fitness ranking (not value)
    for key in ['DNA', 'mut_strength']:
        pop[key] = pop[key][good_idx]
    return pop, fitness[good_idx]

def write_header_txt(file):
    file.write('------------------------------------ \n')
    file.write('PARAMETERS used: \n')
    file.write(' - POPULATION SIZE (parents): %r\n' % str(POPULATION_SIZE))
    file.write(' - NUMBER OF KIDS: %r\n' % str(N_KID))
    file.write(' - SIZE TOURNAMENT: %r\n' % str(size_tournament))
    file.write(' - DNA SIZE: %r\n' % str(DNA_SIZE))
    file.write('------------------------------------ \n\n\n')
    file.write('------------------------------------ \n')
    file.write('Fitness_value\t\tIndividual\n')
    file.write('------------------------------------ \n')

def main():
    global previous_fitness_pop
    name = str(save_results+'.txt')
    file = open(name, "w")
    write_header_txt(file)

    # initialize randomly the population DNA values (motor angles)
    # initialize randomly the variances with big values
    population = dict(DNA=np.random.uniform(low=DNA_BOUND[0], high=DNA_BOUND[1], size=(POPULATION_SIZE,DNA_SIZE) ),   
            mut_strength=np.random.rand(POPULATION_SIZE, DNA_SIZE)*10)                                     
            # mut_strength=np.random.rand(POPULATION_SIZE, DNA_SIZE)*(DNA_BOUND[1]/2))                                     
    
    # for plotting
    if PLOTTING_REAL_TIME == 1:
        plt.plot(generations_plt, fitness_curve, 'b', linewidth=1.0, label='Best individual fitness')
        plt.plot(generations_plt, fitness_curve2, 'r', linewidth=1.0, label='Second best individual fitness')
        plt.plot(generations_plt, fitness_curve3, 'g', linewidth=1.0, label='Third best individual fitness')
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.legend()
    
    # create Session with server
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=POPULATION_SIZE, pool_maxsize=POPULATION_SIZE, max_retries=MAX_RETRIES)
    session.mount('http://', adapter)

    for i in range(N_GENERATIONS):
        kids = make_kid(population, N_KID)
        population, fitness_pop = kill_bad(population, kids, session)   # keep some good parent for elitism
        # fitness_population = evaluation(population)

        fitness_curve.append(fitness_pop[0]) # Append best individual for plotting

        
        improvements = np.isclose(fitness_pop,previous_fitness_pop,atol=0.01)

        for j in range(POPULATION_SIZE):
            if (improvements[j] == True):
                population['mut_strength'][j] = population['mut_strength'][j] * c # decrease mutation



        previous_fitness_pop = fitness_pop


        # ------------------ PLOTTING, DRAWING AND WRITING... ----------------------------

        if (fitness_curve[i-1] != fitness_curve[i]):
            file.write(' %r\t\t\t%r\n\t\t\t\t\t\t%r\n\n' % (fitness_pop[0],population['DNA'][0],population['mut_strength'][0]))

        if PLOTTING_REAL_TIME == 1:
            generations_plt.append(i)
            unique = np.unique(fitness_pop) # return ordered unique population fitness
            fitness_curve2.append(unique[1]) # second best individual
            fitness_curve3.append(unique[2]) # third best individual
            plt.plot(generations_plt, fitness_curve, 'b', linewidth=1.0, label='Best individual fitness')
            plt.plot(generations_plt, fitness_curve2, 'r', linewidth=1.0, label='Second best individual fitness')
            plt.plot(generations_plt, fitness_curve3, 'g', linewidth=1.0, label='Third best individual fitness')
            plt.pause(0.001)

        if i%10==0:
            print("Iteration num: ",i)
            print("  Best fitness value: ",fitness_pop[0])
            print("    - DNA value: ",population['DNA'][0])
            print("    - Mutation values: ",population['mut_strength'][0])
            print("  Standard deviation population DNA: ",np.std(population['DNA']))
            print("  Standard deviation population mut_strength: ",np.std(population['mut_strength']))
            print("  Num of different individuals: ",len(unique))
            print("-------------------------------")

        if (fitness_pop[0] == 0 or i == N_GENERATIONS-1):
            plt.savefig(str(save_results+'.png'))
            file.write('------------------------------------ \n\n\n')
            file.write('------------------------------------ \n')
            file.write('TOTAL GENERATIONS: %r\n' % (i+1))
            file.write('------------------------------------ \n')
            file.close()
            if fitness_pop[0] == 0:
                print(" => Optimal solution has been found!")
            else:
                print(" *** Maximum number of iterations reached ***")
            break




if __name__ == "__main__":
    main()