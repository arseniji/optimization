import numpy as np

POP_SIZE = 100
N_GENERATIONS = 300
BOUNDS = (-2, 2)
MUTATION_RATE = 0.1
MUTATION_STD = 0.1
CROSSOVER_RATE = 0.8
ELITE_SIZE = 5
SEED = 42

rng = np.random.default_rng(SEED)


def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def init_population(size, bounds):
    lo, hi = bounds
    return rng.uniform(lo, hi, size=(size, 2))


def fitness(population):
    return rosenbrock(population[:, 0], population[:, 1])


def tournament_selection(population, fit_values, tournament_size=3):
    parents = []
    for _ in range(2):
        idx = rng.integers(0, len(population), size=tournament_size)
        winner = idx[np.argmin(fit_values[idx])]
        parents.append(population[winner])
    return parents[0], parents[1]


def crossover(parent1, parent2):
    if rng.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()
    alpha = rng.random()
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = alpha * parent2 + (1 - alpha) * parent1
    return child1, child2


def mutate(individual):
    individual = individual.copy()
    for i in range(len(individual)):
        if rng.random() < MUTATION_RATE:
            individual[i] += rng.normal(0, MUTATION_STD)
            individual[i] = np.clip(individual[i], *BOUNDS)
    return individual


def genetic_algorithm():
    population = init_population(POP_SIZE, BOUNDS)

    history_best = []
    history_mean = []
    history_point = []

    for gen in range(N_GENERATIONS):
        fit_values = fitness(population)

        best_idx = np.argmin(fit_values)
        history_best.append(fit_values[best_idx])
        history_mean.append(np.mean(fit_values))
        history_point.append(population[best_idx].copy())

        elite_idx = np.argsort(fit_values)[:ELITE_SIZE]
        new_population = [population[i].copy() for i in elite_idx]

        while len(new_population) < POP_SIZE:
            p1, p2 = tournament_selection(population, fit_values)
            c1, c2 = crossover(p1, p2)
            new_population.append(mutate(c1))
            if len(new_population) < POP_SIZE:
                new_population.append(mutate(c2))

        population = np.array(new_population)

    fit_values = fitness(population)
    best_idx = np.argmin(fit_values)
    best_x, best_y = population[best_idx]
    best_f = fit_values[best_idx]

    return best_x, best_y, best_f, history_best, history_mean, history_point

best_x, best_y, best_f, history_best, history_mean, history_point = genetic_algorithm()

print("=" * 45)
print("  РЕЗУЛЬТАТ ГЕНЕТИЧЕСКОГО АЛГОРИТМА")
print("=" * 45)
print(f"  Найденная точка : x = {best_x:.6f}, y = {best_y:.6f}")
print(f"  Значение f(x,y) : {best_f:.8f}")
print(f"  Теоретический минимум: (1, 1), f = 0")
print(f"  Отклонение от минимума: {np.sqrt((best_x - 1) ** 2 + (best_y - 1) ** 2):.6f}")
print("=" * 45)

