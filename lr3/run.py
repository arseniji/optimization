import numpy as np

POP_SIZE = 100
N_GENERATIONS = 300
MUTATION_RATE = 0.1
MUTATION_STD_K = 0.05
CROSSOVER_RATE = 0.8
ELITE_FRAC = 0.05
TOURNAMENT_K = 3


def init_population(pop_size: int, bounds: tuple) -> np.ndarray:
    lo, hi = bounds
    return np.random.uniform(lo, hi, size=(pop_size, 2))

def evaluate_fitness(f, population: np.ndarray) -> np.ndarray:
    return np.array([f(ind[0], ind[1]) for ind in population])

def tournament_selection(
    population: np.ndarray,
    fit_values: np.ndarray,
    tournament_size: int = TOURNAMENT_K
) -> np.ndarray:

    idx = np.random.randint(0, len(population), size=tournament_size)
    winner = idx[np.argmin(fit_values[idx])]
    return population[winner]


def crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    crossover_rate: float = CROSSOVER_RATE
) -> tuple[np.ndarray, np.ndarray]:

    if np.random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()

    alpha = np.random.random()
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = alpha * parent2 + (1 - alpha) * parent1
    return child1, child2


def mutate(
    individual: np.ndarray,
    bounds: tuple,
    mutation_rate: float = MUTATION_RATE,
    mutation_std_k: float = MUTATION_STD_K
) -> np.ndarray:

    lo, hi = bounds
    individual = individual.copy()
    std = (hi - lo) * mutation_std_k

    for i in range(len(individual)):
        if np.random.random() < mutation_rate:
            individual[i] += np.random.normal(0, std)
            individual[i] = np.clip(individual[i], lo, hi)

    return individual


def run_genetic_algorithm(
    f,
    bounds: tuple,
    pop_size: int = POP_SIZE,
    n_gen: int = N_GENERATIONS,
    mut_rate: float = MUTATION_RATE,
    mut_std_k: float = MUTATION_STD_K,
    crossover_rate: float = CROSSOVER_RATE,
    elite_frac: float = ELITE_FRAC,
    tournament_k: int = TOURNAMENT_K,
) -> dict:

    population = init_population(pop_size, bounds)
    elite_size = max(1, int(pop_size * elite_frac))

    history_best = []
    history_mean = []
    history_points = []

    for gen in range(n_gen):

        fit_values = evaluate_fitness(f, population)

        best_idx = np.argmin(fit_values)
        history_best.append(float(fit_values[best_idx]))
        history_mean.append(float(np.mean(fit_values)))
        history_points.append(population[best_idx].copy())

        elite_idx    = np.argsort(fit_values)[:elite_size]
        new_population = [population[i].copy() for i in elite_idx]

        while len(new_population) < pop_size:
            p1 = tournament_selection(population, fit_values, tournament_k)
            p2 = tournament_selection(population, fit_values, tournament_k)

            c1, c2 = crossover(p1, p2, crossover_rate)

            new_population.append(mutate(c1, bounds, mut_rate, mut_std_k))
            if len(new_population) < pop_size:
                new_population.append(mutate(c2, bounds, mut_rate, mut_std_k))

        population = np.array(new_population)

    fit_values = evaluate_fitness(f, population)
    best_idx   = np.argmin(fit_values)

    return {
        "best_x": float(population[best_idx][0]),
        "best_y": float(population[best_idx][1]),
        "best_f": float(fit_values[best_idx]),
        "history_best": history_best,
        "history_mean": history_mean,
        "history_points": np.array(history_points),
    }