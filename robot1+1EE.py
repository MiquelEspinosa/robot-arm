import numpy as np
import requests
import time
import matplotlib.pyplot as plt
import concurrent.futures as futures
from threading import Thread

# NETWORKING
URL_PATH = "http://163.117.164.219/age/robot10?"
MAX_RETRIES = 10

# CONSTANTES
DNA_SIZE = 10                             # Corresponds to the number of motors
DNA_BOUND = [-180, 180]                  # Upper and lower bounds for DNA values
N_GENERATIONS = 2000

# one-fifth rule
c = 0.8                                 # constante para la regla de 1/5
s = 15                                   # tamaño ventana para array de mejoras
success = np.zeros(s)                 # guardamos el número de mejoras por cada s iteraciones

# Plotting
PLOTTING_REAL_TIME = 1  # Choose to show fitness plot in real time
generations_plt = []    # Plotting axis
fitness_curve = []      # Plotting curve
save_results = 'output'

def evaluation(individual, session):
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
        print("Exception2 when calling web service")
        time.sleep(1)
        return evaluation(individual, session)
    return float(r)
   
def make_kid(dad):
    # generate empty kid holder
    kid = {'DNA': np.empty((DNA_SIZE))}
    kid['mut_strength'] = np.array(dad['mut_strength'])
    # DNA mutation 
    kid['DNA'] = np.random.normal(dad['DNA'],dad['mut_strength'])
    # clip the mutated DNA value
    kid['DNA'] = np.clip(kid['DNA'], *DNA_BOUND)
    return kid

def regla1_5(first_iter, mut_values, ratio):
    if first_iter == True:
        return mut_values / c

    if (ratio < (1/5)):
        return mut_values * c
    elif (ratio > (1/5)):
        return mut_values / c
    else:
        return mut_values

def kill_bad(dad, kid, session, num_decimals):
    fit_dad = evaluation(dad['DNA'], session)
    fit_kid = evaluation(kid['DNA'], session)
    if round(fit_dad, num_decimals) < round(fit_kid, num_decimals): 
        return dad, fit_dad, 0 # we did not improve
    else:
        return kid, fit_kid, 1 # we improved

def write_header_txt(file):
    file.write('------------------------------------ \n')
    file.write('PARAMETERS used: \n')
    file.write(' - DNA SIZE: %r\n' % str(DNA_SIZE))
    file.write('------------------------------------ \n\n\n')
    file.write('------------------------------------ \n')
    file.write('Fitness_value\t\tIndividual\n')
    file.write('------------------------------------ \n')

def main():
    name = str(save_results+'.txt')
    file = open(name, "w")
    write_header_txt(file)

    # initialize randomly the population DNA values (motor angles)
    # initialize randomly the variances with big values
    dad = dict(DNA=np.random.uniform(low=DNA_BOUND[0], high=DNA_BOUND[1], size=DNA_SIZE ),   
            mut_strength=np.random.rand(DNA_SIZE)*10)                                     
    
    # for plotting
    if PLOTTING_REAL_TIME == 1:
        plt.plot(generations_plt, fitness_curve, 'b', linewidth=1.0, label='Best individual fitness')
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.legend()
    
    # create Session with server
    session = requests.Session()

    for i in range(N_GENERATIONS):
        
        # small trick for fast convergence at the start, and slower and precise at the end
        if (i < (N_GENERATIONS*0.8)):
            num_decimals = 0
        elif (i < (N_GENERATIONS*0.9)):
            num_decimals = 1
        else:
            num_decimals = 2

        kid = make_kid(dad)
        best, fitness, winner = kill_bad(dad, kid, session, num_decimals)
        
        j = i%s # posición en array "success"
        success[j] = winner # winner is 0 if dad won, or 1 if kid won
        ratio = np.sum(success)/s # ratio de aciertos en las ultimas s iteraciones

        if i<c: best['mut_strength'] = regla1_5(True, best['mut_strength'], ratio)
        else: best['mut_strength'] = regla1_5(False, best['mut_strength'], ratio)

        fitness_curve.append(fitness) # Append best individual for plotting

        dad = best


        # ------- PLOTTING, PRINTING AND OTHERS ---------------

        if (fitness_curve[i-1] != fitness_curve[i]):
            file.write(' %r\t\t\t%r\n\t\t\t\t\t\t%r\n\n' % (fitness,best['DNA'],best['mut_strength']))

        if PLOTTING_REAL_TIME == 1:
            generations_plt.append(i)
            plt.plot(generations_plt, fitness_curve, 'b', linewidth=1.0, label='Best individual fitness')
            plt.pause(0.0001)

        if i%10==0:
            print("Iteration num: ",i)
            print("    - Fitness value: ",fitness)
            print("    - DNA value: ",best['DNA'])
            print("    - Mutation values: ",best['mut_strength'])
            print("-------------------------------")

        if (fitness == 0 or i == N_GENERATIONS-1):
            plt.savefig(str(save_results+'.png'))
            file.write('------------------------------------ \n\n\n')
            file.write('------------------------------------ \n')
            file.write('TOTAL GENERATIONS: %r\n' % (i+1))
            file.write('------------------------------------ \n')
            file.close()
            if fitness == 0:
                print(" => Optimal solution has been found!")
            else:
                print(" *** Maximum number of iterations reached ***")
            break





if __name__ == "__main__":
    main()