import random
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

def penalizacion(individual):
    suma = sum(individual)

    if suma != 10:
        return True
    else:
        return False

def aptitudIndividuo(individual):
    c1, c2, c3, c4 = [individual[i] for i in (0, 1, 2, 3)]

    dividendo = (c1 + c2 + c3 + c4)
    v = abs(dividendo - 10)
    
    validaPenalizacion = penalizacion(individual)
    aptitud = dividendo / (500 * v + 1)
    aptitud = aptitud if not validaPenalizacion else 0

    return aptitud

creator.create('FitnessMax', base.Fitness, weights = (1.0,))
creator.create('individual', list, fitness = creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('attr_bool', random.randint, 0, 1)
toolbox.register('individual', tools.initRepeat, creator.individual, toolbox.attr_bool, 4)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

toolbox.register('evaluate', aptitudIndividuo)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)
toolbox.register('select', tools.selTournament, tournsize=2)

def programa():
    numeroPoblacion      = int(input("Numero de individuos para la poblacion: "))
    probabilidadCruza    = float(input("Probabilidad de cruza: "))
    probabilidadMutacion = float(input("Probabilidad de mutacion: "))
    numeroGeneraciones   = int(input("Numero de generaciones: "))

    poblacion            = toolbox.population(n=numeroPoblacion)
    hof                  = tools.HallOfFame(1)
    stats                = tools.Statistics(lambda ind: ind.fitness.values)
    
    stats.register('avg', np.mean)
    stats.register('std', np.std)
    stats.register('min', np.min)
    stats.register('max', np.max)

    algorithms.eaSimple(poblacion, toolbox, cxpb=probabilidadCruza, mutpb=probabilidadMutacion, ngen=numeroGeneraciones, stats=stats, halloffame=hof)

    return 0


if __name__ == "__main__":
    prog = programa()