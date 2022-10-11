import sys
import numpy as np
def fitness(equation_points, equation_inputs):
    # calcular o fitness da população atual
    # calcula a soma dos produtos de w.x (população x inputs)

    soma = np.sum(equation_inputs * equation_points, axis=1)

    fitness = []

    for i in soma :
        if (i > 30 ) :
            i *= -9999999

        fitness.append(i)

    return np.array(fitness)

def selection(population, fitness, num_parents):
    # Selecionar os melhores indivíduos para o grupo de
    # cruzamento
    # cria um vetor vazio com o tamanho do número de genitores
    parents = np.empty((num_parents, population.shape[1])) # 4 x 6
    # preenche o vetor de genitores
    for idx in range(num_parents):
        # obter o índice do elemento com o maior fitness
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[idx, :] = population[max_fitness_idx, :]
        fitness[max_fitness_idx] = -999999
    return parents

def crossover(parents, generation_size):
    # gera o crossover entre os genitores
    # os filhos terão um tamanho de h linhas por w colunas
    # parents: numpy array contendo os genitores
    # generation_size: tupla de (h, w)
    # cria o vetor de "filhos" vazio
    offspring = np.empty(generation_size)
    # obtém o ponto de corte (crossover)
    # obtém o índice que é a metade do tamanho do cromossomo
    # (metade do número de colunas)
    crossover_point = np.uint8(generation_size[1]/2)
    # iterar pelos genitores para gerar a prole
    # itera pela quantidade de filhos a serem gerados
    for idx in range(generation_size[0]):
        # índice do primeiro genitor
        p1_idx = idx % parents.shape[0]
        # índice do segundo genitor
        p2_idx = (idx + 1) % parents.shape[0]
        # o novo filho terá a primeira metade de seus genes
        # oriundos do primeiro genitor
        offspring[idx, 0:crossover_point] = parents[
            p1_idx, 0:crossover_point]
        # o novo filho terá a segunda metade de seus genes
        # oriundos do segundo genitor
        offspring[idx, crossover_point:] = parents[
            p2_idx, crossover_point:
        ]
    return offspring

def mutation(offspring):
    # altera um gene aleatório em cada filho
    for idx in range(offspring.shape[0]):
        # gerar um valor aleatório para adicionar ao filho

        # random_value = np.random.uniform(-1.0, 1.0, 1)

        # obtém um gene aleatório (um valor entre 0 e o número de colunas)
        random_idx = np.random.randint(offspring.shape[1])
        # altera o gene do random_idx do filho idx
        offspring[idx, random_idx] = (
            abs(offspring[idx, random_idx] - 1)
            # offspring[idx, random_idx]) + random_value
        )
    return offspring