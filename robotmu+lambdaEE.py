import numpy as np
import requests
import time
import matplotlib.pyplot as plt
import concurrent.futures as futures

# lamda = kids
# mu = progenitores

# NETWORKING
URL_PATH = "http://163.117.164.219/age/robot10b?"
MAX_RETRIES = 10


DNA_SIZE = 10                             # Corresponds to the number of motors
DNA_BOUND = [-180, 180]                  # Upper and lower bounds for DNA values
POPULATION_SIZE = 600                     # Population size
N_GENERATIONS = 1000
N_KID = 450                               # n kids per generation = lambda
tau = 1/np.sqrt(2*np.sqrt(DNA_SIZE))     # consideramos b = 1
tau0 = 1/np.sqrt(2*DNA_SIZE)             # consideramos b = 1
size_tournament = 4


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
    return abs(float(r))

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

def tournament(population, size_tournament, num_parents, fitness_pop):
    selected = [] # list of indexes of all tournament winners
    for _ in range(num_parents):
        tournament_fitness = []
        participants = np.random.choice(np.arange(POPULATION_SIZE), size=size_tournament, replace=False)
        for j in participants:
            tournament_fitness.append(fitness_pop[j])
        index = np.argmin(tournament_fitness) # get index of min fitness
        selected.append(participants[index]) # append index of participant with min fitness
    return selected
    
def make_kid(population, n_kid, fitness_pop):
    # generate empty kid holder
    kids = {'DNA': np.zeros((n_kid, DNA_SIZE)),
            'mut_strength': np.zeros((n_kid, DNA_SIZE))}

    for i in range(n_kid):
        variances = np.zeros(DNA_SIZE) # initialize array
        dna = np.zeros(DNA_SIZE) # initialize array        
        parents = tournament(population['DNA'], size_tournament, 3, fitness_pop)
        
        # uniform crossing with average among parents
        dna = (population['DNA'][parents[0]] + population['DNA'][parents[1]] + population['DNA'][parents[2]]) / 3
        # cruce posicional de varianzas
        for j in range(DNA_SIZE):
            variances[j] = population['mut_strength'][parents[np.random.randint(3)]][j]

        # DNA and variances mutation 
        dna = np.random.normal(dna,variances)
        random_tau = np.exp(np.random.normal(0,tau,DNA_SIZE))
        random_tau0 = np.exp(np.random.normal(0,tau0,DNA_SIZE))
        variances = variances * random_tau * random_tau0

        # clip the mutated value
        dna[:] = np.clip(dna, *DNA_BOUND)

        # update kids array
        kids['DNA'][i] = dna
        kids['mut_strength'][i] = variances

    return kids


def kill_bad(pop, kids, session, fitness_dads):
    # put population and kids together
    for key in ['DNA', 'mut_strength']:
        pop[key] = np.vstack((pop[key], kids[key]))
    
    fitness_kids = evaluation(kids, session)            # calculate global fitness
    fitness = np.concatenate((fitness_dads, fitness_kids))
        
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

    fitness_pop = evaluation(population, session)            # calculate global fitness
    
    for i in range(N_GENERATIONS):
        kids = make_kid(population, N_KID, fitness_pop)
        population, fitness_pop = kill_bad(population, kids, session, fitness_pop)   # keep some good parent for elitism
        fitness_curve.append(fitness_pop[0]) # Append best individual for plotting
        
        if np.std(population['mut_strength']) < 0.01:
            population['mut_strength'] = population['mut_strength'] * 10

        # ------------------ PLOTTING, DRAWING AND WRITING... ----------------------------

        if (fitness_curve[i-1] != fitness_curve[i]):
            file.write(' %r\t\t\t%r\n\t\t\t\t\t\t%r\n\n' % (fitness_pop[0],population['DNA'][0],population['mut_strength'][0]))

        if PLOTTING_REAL_TIME == 1:
            generations_plt.append(i)
            unique = np.unique(fitness_pop) # return ordered unique population fitness
            plt.plot(generations_plt, fitness_curve, 'b', linewidth=1.0, label='Best individual fitness')
            if i < N_GENERATIONS/2:
                if len(unique)>1:
                    fitness_curve2.append(unique[1]) # second best individual
                    plt.plot(generations_plt, fitness_curve2, 'r', linewidth=1.0, label='Second best individual fitness')
                if len(unique)>2:
                    fitness_curve3.append(unique[2]) # third best individual
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