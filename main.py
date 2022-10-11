# Y = w1.x1 + w2.x2 + w3.x3 + w4.x4 + w5.x5 + w6.x6
import numpy as np
import ga
def main():
    # Valores para as variáveis xi
    equation_inputs = [15, 3, 2, 5, 9, 20]
    equation_points = [17, 7, 10, 5, 8, 17]   # new ArrayList<>();
    # Número de pesos (w) a otimizar
    num_weights = 6  # número de genes
    # tamanho da população
    solutions_per_equation_points = 6  # número de cromossomos
    # um conjunto de 8 x 6
    equation_points_size = (solutions_per_equation_points, num_weights)
    # Criar população inicial (inicializar)
    equation_points = np.random.randint(
        low=0, high=2,
        size=equation_points_size)
    
    print("População inicial:")
    print(equation_points)
    # número de gerações
    num_generations = 5
    # número de genitores para cruzamento
    num_parents_crossover = 4
    # para cada geração
    # for (int i = 0, i < num_generations; i++)
    for generation in range(num_generations):
        print(f"\nGeração {generation}")
        # calcular o fitness
        fitness = ga.fitness(equation_inputs, equation_points)
        print("\nFitness:")
        print(fitness)
        # selecionar os melhores indivíduos
        selected_parents = ga.selection(
            equation_points, fitness, num_parents_crossover)
        print("\nGenitores selecionados:")
        print(selected_parents)
        # fazer o crossover entre os melhores indivíduos
        offspring_crossover = ga.crossover(
            selected_parents, (
                solutions_per_equation_points - num_parents_crossover,
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
        equation_points[0:selected_parents.shape[0], :] = selected_parents
        # crossover + mutação
        equation_points[selected_parents.shape[0]:, :] = offspring_mutation
        print("\nNova população:")
        print(equation_points)
        print("Melhor resultado: ", np.max(
            ga.fitness(equation_inputs, equation_points)))
    fitness = ga.fitness(equation_inputs, equation_points)
    best_fit_idx = np.where(fitness == np.max(fitness))
    print("Melhor resultado: ", equation_points[best_fit_idx, :])
    print("Fitness do melhor: ", fitness[best_fit_idx])

if __name__ == "__main__":
    main()