# Y = w1.x1 + w2.x2 + w3.x3 + w4.x4 + w5.x5 + w6.x6
import numpy as np
import ga
def main():
    # Valores para as variáveis xi
    equation_inputs = [15, 10, 5, 5, 8, 17]  # new ArrayList<>();
    # Número de pesos (w) a otimizar
    num_weights = 6  # número de genes
    # tamanho da população
    solutions_per_population = 6  # número de cromossomos
    # um conjunto de 8 x 6
    population_size = (solutions_per_population, num_weights)
    # Criar população inicial (inicializar)
    population = np.random.randint(
        low=0, high=2,
        size=population_size)
    
    print("População inicial:")
    print(population)
    # número de gerações
    num_generations = 5
    # número de genitores para cruzamento
    num_parents_crossover = 4
    # para cada geração
    # for (int i = 0, i < num_generations; i++)
    for generation in range(num_generations):
        print(f"\nGeração {generation}")
        # calcular o fitness
        fitness = ga.fitness(equation_inputs, population)
        print("\nFitness:")
        print(fitness)
        # selecionar os melhores indivíduos
        selected_parents = ga.selection(
            population, fitness, num_parents_crossover)
        print("\nGenitores selecionados:")
        print(selected_parents)
        # fazer o crossover entre os melhores indivíduos
        offspring_crossover = ga.crossover(
            selected_parents, (
                solutions_per_population - num_parents_crossover,
                num_weights
            )
        )
        print("\nFilhos gerados por crossover:")
        print(offspring_crossover)
        # adicionar mutação nos filhos gerados
        offspring_mutation = ga.mutation(offspring_crossover)
        print("\nFilhos pós mutação:")
        print(offspring_mutation)
        # criar a nova população
        # elitismo
        population[0:selected_parents.shape[0], :] = selected_parents
        # crossover + mutação
        population[selected_parents.shape[0]:, :] = offspring_mutation
        print("\nNova população:")
        print(population)
        print("Melhor resultado: ", np.max(
            ga.fitness(equation_inputs, population)))
    fitness = ga.fitness(equation_inputs, population)
    best_fit_idx = np.where(fitness == np.max(fitness))
    print("Melhor resultado: ", population[best_fit_idx, :])
    print("Fitness do melhor: ", fitness[best_fit_idx])

if __name__ == "__main__":
    main()