import random
import string
import matplotlib.pyplot as plt
target = "11111111111111110000000000000000"
best = []
def main(populationSize, chromosomeSize, genBounds, mutRate, crossRate, maxGens, graph):

    population = createPopulation(populationSize, chromosomeSize, genBounds)
    # Population Created 
    # print(population)

    solution = []

    for gen in range(maxGens):
        # How many generations we are iterating through
        # population = selection(population)

        if target in population:
            solution = target
            break
        # Crossover
        crossoverGen = []
        for j in range(0, populationSize, 2):
            parent1 = population[j]
            parent2 = population[j+1]
            if random.random() < crossRate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            crossoverGen.append(child1)
            crossoverGen.append(child2)


        # Mutation
        mutatedGen = []
        for individual in population:
            mutatedIndividual = mutation(individual, mutRate, genBounds)
            mutatedGen.append(mutatedIndividual)
        population = binaryTournamentSelection(population, crossoverGen, mutatedGen, gen)
        
        # if target found in population
        if target in population:
            solution = target
            print("Solution found:", solution)
            if graph:  
                plt.plot([x[0] for x in best], [x[1] for x in best])
                plt.title("Best Fitness out of population over Generations")
                plt.xlabel("Generation")
                plt.ylabel("Fitness")
                plt.show()
            break


    return solution

def createPopulation(populationSize, chromosomeSize, genBounds):
        # creates population
        population = []
        for _i in range(populationSize):
            individual = ""
            for _j in range(chromosomeSize):
                bit = random.randint(0, 1)
                individual += str(bit)
            population.append(individual)
        return population

def fitness(individual):
    x, y = decode(individual)
    # print("Fitness - Decoded X:", x, "Decoded Y:", y)
    return x - y

def binaryTournamentSelection(population, crossGen, mutatedGen, gen):
    """
    Use binary (2-way) tournament selection on the combined candidate pool
    to produce a new population of the same size as `population`.
    """
    k = len(population)
    candidates = population + crossGen + mutatedGen

    # Run binary tournaments to fill the new population 
    selected = []
    for _ in range(k):
        a, b = random.sample(candidates, 2)
        winner = a if fitness(a) >= fitness(b) else b
        selected.append(winner)
    # added best individual so I can print to graph and console
    best_individual = max(candidates, key=fitness)
    print("Best individual selected:", best_individual, "with fitness:", fitness(best_individual))
    best.append((gen, fitness(best_individual)))
    return selected


def mutation(individual, mutRate, genBounds):
    mutated = ""
    for bit in individual:
        if random.random() < mutRate:
            bit = '1' if bit == '0' else '0'
        mutated += bit
    return mutated

def crossover(parent1, parent2):
    crossPoint = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossPoint] + parent2[crossPoint:]
    child2 = parent2[:crossPoint] + parent1[crossPoint:]
    # print("Child1:", child1, "Child2:", child2, "Parent1:", parent1, "Parent2:", parent2)
    return child1, child2

def decode(individual):
    x = int(individual[:16], 2)
    y = int(individual[16:], 2)
    # print("Decoded X:", x, "Decoded Y:", y)
    return x, y

def fullFactorial():
    pass


main(populationSize=100, chromosomeSize=32, genBounds=(0, 1), mutRate=.2, crossRate=0.2, maxGens=150, graph=True)
# population size
# chromosome size
# gen bounds
# mut rate 
# cross rate
# maxGens
# 
# Graph = True/False

''' 
Brief Results and Explanation:

'''