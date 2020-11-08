import random
import requests
import time
import math
import matplotlib.pyplot as plt
import sys, getopt

# Parameters
chromosome_size = 64  # Tamano del cromosoma 64 = 4 estaciones * 16 sensores/estacion
debug_level = 0
max_iterations = 50

# Dynamic options for tournament size
DYNAMIC_T_SIZE = True
MIN_T_SIZE = 0.03
MAX_T_SIZE = 0.2

# Dynamic options for mutation size
DYNAMIC_M_SIZE = True
MIN_M_SIZE = 0.03
MAX_M_SIZE = 0.1

# ---- Default values to override ----
save_results = 'default'
population_size = 100  # Normalmente mayor, tipo 100
percentage_tournament = 0.05  # Con poblaciones de 100 suele ser un 2%-5% depende
percentage_mutation = 0.05
pure_elitism = True

# -------------------------- Arguments parsing -------------------------- #
# Options 
options = "f:h:p:t:m:e:"
# Long options 
long_options = ["file=", "help=", "population=", "tournament=", "mutation=", "elitism="]

try:
    opts, args = getopt.getopt(sys.argv[1:],options,long_options)
except getopt.GetoptError:
    print('main.py -f <outputfile> -h <help> -p <population_size> -t <tournament_size> -m <mutation_size> -e <pure_elitism>')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('main.py -f <outputfile> -h <help> -p <population_size> -t <tournament_size> -m <mutation_size> -e <pure_elitism>')
        sys.exit()
    elif opt in ("-f", "--file"):
        save_results = arg
    elif opt in ("-p", "--population"):
        population_size = int(arg)
    elif opt in ("-t", "--tournament"):
        percentage_tournament = float(arg)
    elif opt in ("-m", "--mutation"):
        percentage_mutation = float(arg)
    elif opt in ("-e", "--elitism"):
        pure_elitism = eval(arg)

# ------------------------------------------------------------------------------ #


# Plotting
PLOTTING_REAL_TIME = 1  # Choose to show fitness plot in real time
generations_plt = []    # Plotting axis
fitness_curve = []      # Plotting curve

def initialization():
    # ------------------------------------------
    # Step 1: Uniform random initialization of all genes/bits
    # ------------------------------------------
    population = []
    for x in range(population_size):
        population.append([])
        chromosome_partial = ''
        for _ in range(chromosome_size):
            randInt = random.randint(0, 1)
            chromosome_partial = chromosome_partial + str(randInt)
        population[x].append(chromosome_partial)
    return population

def get_individual_fitness(individual):
    # This is for getting the fitness of an specific individual
    url = "http://memento.evannai.inf.uc3m.es/age/test?c="
    url = url + str(individual[0])
    try:
        if debug_level >= 2:
            print(str(individual[0]))
        r = requests.get(url).content
    except:
        print("Exception when calling web service")
        time.sleep(1)
        r = requests.get(url).content
        # Podria hacer un bucle con un número máximo de intentos aquí para
        # evitar timeouts y problemas en la llamada al servicio web
    return float(r)


def evaluate_population(population):
    # ------------------------------------------
    # Step 2: This method evaluates a population
    # ------------------------------------------
    population_fitness = []
    best_individual = [float('inf'), None] # List with two elements (min fit value, best individual)
    for x in range(population_size):
        ind_fitness = get_individual_fitness(population[x])
        population_fitness.append(ind_fitness)
        if ind_fitness < best_individual[0]:
            best_individual[0] = ind_fitness
            best_individual[1] = population[x][0]

    return population_fitness, best_individual

def tournament_selection(niter, population, population_fitness):
    # ------------------------------------------
    # Step 3: This method selects best individuals in the population using tournament
    # ------------------------------------------ 
    global percentage_tournament
    # Sigmoid function for a dynamic tournament size
    if DYNAMIC_T_SIZE:
        aux1 = 1+(math.e**(-0.3*(niter-(max_iterations/2))))
        percentage_tournament = MIN_T_SIZE + (MAX_T_SIZE/aux1)
    
    t_size = math.floor(percentage_tournament * population_size)
    print(t_size)

    new_population = []

    if debug_level >= 1:
        print("Tamaño torneo: " + str(t_size))

    for _ in range(population_size):
        selected_index = random.sample(range(population_size), t_size)
        best_fitness_round = float('inf')
        winner = -1  # Index for the winning individual

        for y in selected_index:
            # Select lower or best individual
            if debug_level >=2 :
                print("Tournament participants: " + str(y) + " " + str(population_fitness[y]))             
            if population_fitness[y] < best_fitness_round:
                winner = y
                best_fitness_round = population_fitness[y]

        new_population.append(population[winner])   

    return new_population

def reproduction(population):
    # ------------------------------------------
    # Step 4: Reproduction among individuals in population
    # ------------------------------------------ 
    new_population = []
    for i in range(0,population_size,2):
        padre = population[i][0]
        madre = population[i+1][0]
        son1 = ''
        son2 = ''

        for j in range(chromosome_size):
            rand1 = random.random()
            rand2 = random.random()
            # primer son
            if (rand1 > 0.5):
                son1 = son1 + padre[j]
            else:
                son1 = son1 + madre[j]
            # second son
            if (rand2 > 0.5):
                son2 = son2 + padre[j]
            else:
                son2 = son2 + madre[j]
        
        new_population.append([son1])
        new_population.append([son2])

    return new_population

def mutation(niter, population):
    # ------------------------------------------
    # Step 5: Mutation of individuals given a mutation percentage
    # ------------------------------------------ 
    global percentage_mutation
    # Inverse sigmoid function for a dynamic mutation size
    if DYNAMIC_M_SIZE:
        aux1 = 1+(math.e**(0.3*(niter-(max_iterations/2))))
        percentage_mutation = MIN_M_SIZE + (MAX_M_SIZE/aux1)
    print(percentage_mutation)
    for i in range(population_size):
        for j in range(chromosome_size):
            rand = random.random()
            if rand < percentage_mutation:

                if debug_level > 2:
                    print("MUTATION in position ", j)
                    print("before ",population[i])
                
                num = population[i][0][j]
                individual = list(population[i][0])

                if 0 == int(num):
                    individual[j] = '1'
                else: 
                    individual[j] = '0'

                population[i][0] = "".join(individual)

                if debug_level > 2:
                    print("after  ",population[i])

    return population

def write_header_txt(file):
    file.write('------------------------------------ \n')
    file.write('PARAMETERS used: \n')
    file.write(' - PURE ELITISM: %r\n' % str(pure_elitism))
    file.write(' - GENERATIONAL: %r\n' % str(not(pure_elitism)))
    file.write(' - Population size: %r\n' % population_size)
    if DYNAMIC_T_SIZE:
        file.write(' - Tournament percentage: dynamic\n')
    else:
        file.write(' - Tournament percentage: %r\n' % percentage_tournament)
    if DYNAMIC_M_SIZE:
        file.write(' - Mutation rate: dynamic\n')
    else:
        file.write(' - Mutation rate: %r\n' % percentage_mutation)
    file.write('------------------------------------ \n\n\n')
    file.write('------------------------------------ \n')
    file.write('Fitness_value\t\tIndividual\n')
    file.write('------------------------------------ \n')

def initialize_plot():
    plt.plot(generations_plt, fitness_curve, 'b', linewidth=1.0, label='Best individual fitness')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.legend()


def main():
    # Population initialization
    population = initialization()

    # TXT file for storing best generation individual
    name = str(save_results+'.txt')
    file = open(name, "w")
    write_header_txt(file)
    initialize_plot()

    for i in range(0, max_iterations):
        print("--- Generation ", i," ---")

        # Evaluate population and get best individual
        population_fitness, best_individual = evaluate_population(population)
        generations_plt.append(i)
        fitness_curve.append(best_individual[0])

        # TXT write best individual value
        file.write('  %r\t\t\t%r\n' % (best_individual[0],best_individual[1]))
        
        # Plot best individual
        if PLOTTING_REAL_TIME == 1:
            plt.plot(generations_plt, fitness_curve, 'b', linewidth=1.0, label='Best individual fitness')
            plt.pause(0.05)
        
        # Stopping condition
        if (best_individual[0] == 0 or i == max_iterations-1):
            plt.savefig(str(save_results+'.png'))
            file.write('------------------------------------ \n\n\n')
            file.write('------------------------------------ \n')
            file.write('TOTAL GENERATIONS: %r\n' % (i+1))
            file.write('------------------------------------ \n')
            file.close()
            if best_individual[0] == 0:
                print(" => Optimal solution has been found!")
            else:
                print(" *** Maximum number of iterations reached ***")
            break

        if debug_level >= 1:
            print(population_fitness)
            print("Primer individuo poblacion: " + str(population[0]))
            print("Primer individuo poblacion: " + str(population[population_size-1]))
            print("Tamaño poblacion: " + str(len(population)))
            print("Tamaño cromosoma: " + str(len(population[0][0])))

        new_population = tournament_selection(i, population, population_fitness)
        # print(new_population)
        population_sons = reproduction(new_population)
        # print(population_sons)
        mutated_population = mutation(i,population_sons)
        # print(population)

        time.sleep(1)
        if pure_elitism:
            random_ind = random.randint(0, population_size-1)
            mutated_population[random_ind][0] = best_individual[1]
            # pop_fitn, _ = evaluate_population(mutated_population)
            # max_pos = pop_fitn.index(max(pop_fitn))
            # mutated_population[max_pos][0] = best_individual[1]
        
        population = mutated_population
    
    file.close()


if __name__ == "__main__":
    main()
