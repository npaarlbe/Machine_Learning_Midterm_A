import random
import string
import matplotlib.pyplot as plt
# Nate Paarlberg and Jacob Stetka 
target = "11111111111111110000000000000000"
best = []
global quickestGen
quickestGen = 0
def main(populationSize, chromosomeSize, genBounds, mutRate, crossRate, maxGens, graph, fullFactorial=False):

    population = createPopulation(populationSize, chromosomeSize, genBounds)
    # Population Created 
    # print(population)

    solution = []
    for gen in range(maxGens):

        # if full factorial testing, break when we exceed quickest gen found
        if fullFactorial:
            if gen > quickestGen and quickestGen != 0:
                break
        # How many generations we are iterating through
        # population = selection(population)

        # Crossover
        crossoverGen = []
        # iterate over population in pairs; handle odd population sizes by pairing the last individual with itself
        for j in range(0, len(population), 2):
            parent1 = population[j]
            parent2 = population[j+1] if j+1 < len(population) else population[j]
            if random.random() < crossRate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            crossoverGen.append(child1)
            crossoverGen.append(child2)


        # Mutation (apply to offspring from crossover)
        mutatedGen = []
        for individual in crossoverGen:
            mutatedIndividual = mutation(individual, mutRate, genBounds)
            mutatedGen.append(mutatedIndividual)
        population = binaryTournamentSelection(population, crossoverGen, mutatedGen, gen)
        
        # if target found in population
        if target in population:
            solution = target
            print("Solution found:", solution)
            print("Generation Found on:", gen, "Population Size:", populationSize, "Mutation Rate:", mutRate, "Crossover Rate:", crossRate)
            break

    if graph:  
        plt.plot([x[0] for x in best], [x[1] for x in best])
        plt.title("Best Fitness out of population over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.show()
    if target in population:    
        return solution, gen
    else:
        return solution, None

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
    
    k = len(population)
    candidates = population + crossGen + mutatedGen

    # Run binary tournaments to fill the new population 
    selected = []
    for _ in range(k):
        candidate1, candidate2 = random.sample(candidates, 2)
        bestParent = candidate1 if fitness(candidate1) >= fitness(candidate2) else candidate2
        selected.append(bestParent)
    # added best individual so I can print to graph and console
    best_individual = max(candidates, key=fitness)
    # print("Best individual selected:", best_individual, "with fitness:", fitness(best_individual))
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
    numberOfTests = 0
    global quickestGen
    quickestGen = 0
    writeTable = []
    factorialGraph = []
    
    # Collect data first, then write file
    for j in range(0, 100, 1):
        for k in range(0, 100, 1):
            solution, gen = main(populationSize=100, chromosomeSize=32, genBounds=(0, 1), 
                                mutRate=k/100, crossRate=j/100, maxGens=50, graph=False, fullFactorial=False)
            writeTable.append((k/100, j/100, gen))
            factorialGraph.append((k/100, j/100, gen))
            numberOfTests += 1
            
            # Update quickest generation
            if (gen is None):
                continue
            elif (gen < quickestGen or quickestGen == 0):
                quickestGen = gen
                quickestParams = (100, 32, (0, 1), k/100, j/100, 50)
    
    # Now write file with summary first
    f = open("quickest_params.txt", "w")
    
    # Write blank lines at top
    # for _ in range(50):
    #     f.write("\n")
    
    # Write summary section first
    successful_tests = sum(1 for entry in writeTable if entry[2] is not None)
    f.write("="*80 + "\n")
    f.write("SUMMARY\n")
    f.write("="*80 + "\n")
    f.write(f"Total Tests Run: {numberOfTests}\n")

    
    if quickestGen > 0:
        f.write(f"\nQuickest Generation Found: {quickestGen}\n")
        f.write(f"Best Parameters:\n")
        f.write(f"  - Population Size: {quickestParams[0]}\n")
        f.write(f"  - Chromosome Size: {quickestParams[1]}\n")
        f.write(f"  - Mutation Rate: {quickestParams[3]:.2f}\n")
        f.write(f"  - Crossover Rate: {quickestParams[4]:.2f}\n")
        f.write(f"  - Max Generations: {quickestParams[5]}\n")
    else:
        f.write("\nNo successful runs found.\n")
    
    f.write("\n" + "="*80 + "\n\n")
    f.write("10 QUICKEST GENERATIONS OVERALL:\n")
    f.write("-"*80 + "\n")
    # Sort factorialGraph by generation found, ignoring None values
    sorted_graph = sorted((entry for entry in factorialGraph if entry[2] is not None), key=lambda x: x[2])
    for entry in sorted_graph[:10]:
        f.write(f"Mutation Rate: {entry[0]:.2f}, Crossover Rate: {entry[1]:.2f}, Generation Found: {entry[2]}\n")

    # Then write header and table
    f.write("="*80 + "\n")
    f.write("FULL FACTORIAL ANALYSIS - GENETIC ALGORITHM PARAMETER TESTING\n")
    f.write("="*80 + "\n")
    f.write("Population Size: 100\n")
    f.write("Chromosome Size: 32\n")
    f.write("Max Generations: 50\n")
    f.write("="*80 + "\n\n")
    
    # Write table header
    f.write(f"{'Passed Test #':<8}{'Mutation':<12}{'Crossover':<12}{'Generation':<15}\n")
    f.write("-"*57 + "\n")
    
    # Write successful tests only
    test_num = 1
    for mut, cross, gen in writeTable:
        if gen is not None:
            f.write(f"{test_num:<8}{mut:<12.2f}{cross:<12.2f}{gen:<15}\n")
            test_num += 1
    
    f.write("\n" + "="*80 + "\n")
    f.close()
    
    print("Quickest Generation Found on:", quickestGen, "with parameters:", quickestParams)
    print(f"Full results saved to quickest_params.txt ({numberOfTests} tests)")

# fullFactorial()

print("Running normal test not full factorial")
main(populationSize=1000, chromosomeSize=32, genBounds=(0, 1), mutRate=.02, crossRate=.81, maxGens=150, graph=False)
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

Generation Found on: 5007 Population Size: 10 Mutation Rate: 0.2 Crossover Rate: 0.2 Max Gens: 1500

Generation Found on: 278 Population Size: 100 Mutation Rate: 0.2 Crossover Rate: 0.2 Max Gens: 1500

Generation Found on: 1056 Population Size: 100 Mutation Rate: 0.4 Crossover Rate: 0.2 Max Gens: 1500

Generation Found on: 319 Population Size: 100 Mutation Rate: 0.6 Crossover Rate: 0.6 Max Gens: 1500

Generation Found on: Never Found it Population Size: 100 Mutation Rate: 0.0 Crossover Rate: 0.8 Max Gens: 1500

Generation Found on: Never Found it Population Size: 100 Mutation Rate: 0.8 Crossover Rate: 0.0 Max Gens: 1500

Generation Found on: 156 Population Size: 100 Mutation Rate: 0.8 Crossover Rate: 0.8 Max Gens: 1500

Generation Found on: 101 Population Size: 100 Mutation Rate: 0.99 Crossover Rate: 0.99 Max Gens: 1500

I found that if either mutation or crossover rate was set to 0, the algorithm would often fail to find a solution within the max generations.
This means that both mutation and crossover are important for the genetic algorithm to effectively explore the solution space.

I also found that with a high crossovrer rate it can get it close fast and a low mutation rate can finish the solution

A low population size makes it take much more generations to find a solution, likely due to reduced genetic diversity.

I ran a full factorial with population size 100 and chromosome size 32  and found
Quickest Generation Found: 18
Best Parameters:
  - Population Size: 100
  - Chromosome Size: 32
  - Mutation Rate: 0.02
  - Crossover Rate: 0.81
  - Max Generations: 50


'''