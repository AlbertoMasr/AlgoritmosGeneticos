import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

beneficio = [[0, 0.28, 0.45, 0.65, 0.78, 0.90, 1.02, 1.13, 1.23, 1.32, 1.38],
             [0, 0.25, 0.41, 0.55, 0.65, 0.75, 0.80, 0.85, 0.88, 0.90, 0.90],
             [0, 0.15, 0.25, 0.40, 0.50, 0.62, 0.73, 0.82, 0.90, 0.96, 1.00],
             [0, 0.20, 0.33, 0.42, 0.48, 0.53, 0.56, 0.58, 0.60, 0.60, 0.60]]

def penalizacion(individual):
    suma = sum(individual)

    if suma > 10:
        return True
    else:
        return False

def penalizar(individual):
    suma = sum(individual)

    if suma > 10:
        return "fue penalizado"
    else:
        return "no fue penalizado"

def obtenerResultados(b, inversion):
    return beneficio[b][inversion]

def aptitudIndividuo(individual):
    c1, c2, c3, c4 = [individual[i] for i in (0, 1, 2, 3)]

    c1Resultado = obtenerResultados(0, c1)
    c2Resultado = obtenerResultados(1, c2)
    c3Resultado = obtenerResultados(2, c3)
    c4Resultado = obtenerResultados(3, c4)
    
    dividendo = c1Resultado + c2Resultado + c3Resultado + c4Resultado
    v = abs(dividendo - 10)
    
    validaPenalizacion = penalizacion(individual)
    aptitud = dividendo / (500 * v + 1)
    aptitud = aptitud if not validaPenalizacion else 0

    return aptitud,

def costoIndividuo(individual):
    c1, c2, c3, c4 = [individual[i] for i in (0, 1, 2, 3)]

    c1Resultado = obtenerResultados(0, c1)
    c2Resultado = obtenerResultados(1, c2)
    c3Resultado = obtenerResultados(2, c3)
    c4Resultado = obtenerResultados(3, c4)
    
    costo = c1Resultado + c2Resultado + c3Resultado + c4Resultado

    return costo

creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('individual', list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('attr_bool', random.randint, 0, 10)
toolbox.register('individual', tools.initRepeat, creator.individual, toolbox.attr_bool, 4)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

toolbox.register('evaluate', aptitudIndividuo)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.01)
toolbox.register('select', tools.selTournament, tournsize=2)

def programa():
    numeroPoblacion      = int(input("Numero de individuos para la poblacion: "))
    probabilidadCruza    = float(input("Probabilidad de cruza: "))
    probabilidadMutacion = float(input("Probabilidad de mutacion: "))
    numeroGeneraciones   = int(input("Numero de generaciones: "))

    poblacion            = toolbox.population(n=numeroPoblacion)
    hof                  = tools.HallOfFame(1)
    stats                = tools.Statistics(lambda ind: ind.fitness.values)
    
    print("Poblacion inicial: ")
    
    for elemento in poblacion:
        print("El individuo ", elemento, penalizar(elemento))
    print(" ")

    stats.register('avg', np.mean)
    stats.register('std', np.std)
    stats.register('min', np.min)
    stats.register('max', np.max)

    algorithms.eaSimple(poblacion, toolbox, cxpb=probabilidadCruza, mutpb=probabilidadMutacion, ngen=numeroGeneraciones, stats=stats, halloffame=hof)

    return poblacion, stats, hof


if __name__ == "__main__":
    poblacion, stats, hof = programa()
    print("El mejor individuo fue ", hof, " y tiene un costo de ", costoIndividuo(hof[0]))
