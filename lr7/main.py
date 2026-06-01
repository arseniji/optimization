import numpy as np
from typing import Callable, Tuple


def run_bacterial_optimization(
    f: Callable,
    bounds: Tuple[float, float],
    dims: int = 2,
    pop_size: int = 50,
    n_chemotaxis: int = 50,
    n_reproduction: int = 4,
    n_elimination: int = 2,
    n_swim: int = 4,
    step_size: float = 0.1,
    p_eliminate: float = 0.25,
) -> dict:
    assert pop_size % 2 == 0, "pop_size должен быть чётным"
    lo, hi = bounds

    def fitness(pos: np.ndarray) -> float:
        return -f(pos[0], pos[1])

    pop = np.random.uniform(lo, hi, (pop_size, dims))

    global_best_val = -np.inf
    global_best_pos = pop[0].copy()

    history_best: list = []
    swarm_snapshots: list = []

    for l in range(n_elimination):
        for r in range(n_reproduction):
            health = np.zeros(pop_size)

            for t in range(n_chemotaxis):
                for i in range(pop_size):
                    V = np.random.uniform(-1, 1, dims)
                    norm = np.linalg.norm(V)
                    if norm > 0:
                        V = V / norm

                    cur_fit = fitness(pop[i])
                    health[i] += cur_fit

                    for _ in range(n_swim):
                        new_pos = np.clip(pop[i] + step_size * V, lo, hi)
                        new_fit = fitness(new_pos)
                        if new_fit > cur_fit:
                            pop[i] = new_pos
                            cur_fit = new_fit
                            health[i] += cur_fit
                        else:
                            break

                    if cur_fit > global_best_val:
                        global_best_val = cur_fit
                        global_best_pos = pop[i].copy()

                history_best.append(float(global_best_val))
                swarm_snapshots.append({
                    "xs": pop[:, 0].tolist(),
                    "ys": pop[:, 1].tolist() if dims >= 2 else [0.0] * pop_size,
                    "gbest_x": float(global_best_pos[0]),
                    "gbest_y": float(global_best_pos[1]) if dims >= 2 else 0.0,
                    "gbest_val": float(global_best_val),
                    "iter": l * n_reproduction * n_chemotaxis + r * n_chemotaxis + t,
                })

            sorted_idx = np.argsort(health)[::-1]
            survivors = pop[sorted_idx[: pop_size // 2]]
            pop = np.vstack([survivors, survivors.copy()])

        for i in range(pop_size):
            if np.random.rand() < p_eliminate:
                pop[i] = np.random.uniform(lo, hi, dims)

    return {
        "f": -float(global_best_val),
        "x": global_best_pos.tolist(),
        "iterations": len(history_best),
        "history_best": history_best,
        "swarm_snapshots": swarm_snapshots,
    }