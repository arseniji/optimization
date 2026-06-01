import numpy as np


def run_immune_network(
    f,
    bounds,
    pop_size: int = 50,
    n_b: int = 10,
    n_c: int = 5,
    n_d: int = 5,
    alpha: float = 0.1,
    b_n: float = 0.2,
    max_iter: int = 300,
    stagnation_limit: int = 30,
):
    if isinstance(bounds, (tuple, list)):
        lo, hi = bounds[0], bounds[1]
    else:
        lo, hi = -bounds, bounds

    def fitness(pop):
        return np.array([f(p[0], p[1]) for p in pop])

    S_b = np.random.uniform(lo, hi, (pop_size, 2))
    fit_b = fitness(S_b)

    history_best = []
    history_points = []
    swarm_snapshots = []

    best_val = np.inf
    stag_count = 0

    for iteration in range(max_iter):
        
        idx_sorted = np.argsort(fit_b)
        selected = S_b[idx_sorted[:n_b]].copy()

        
        S_m = []
        for ab in selected:
            clones = np.tile(ab, (n_c, 1))
            clones += alpha * np.random.uniform(-0.5, 0.5, clones.shape)
            clones = np.clip(clones, lo, hi)
            S_m.append(clones)
        S_m = np.vstack(S_m)

        
        fit_m = fitness(S_m)
        idx_m = np.argsort(fit_m)
        S_m_best = S_m[idx_m[:n_d]]
        fit_m_best = fit_m[idx_m[:n_d]]

        
        S_b = np.vstack([S_b, S_m_best])
        fit_b = np.concatenate([fit_b, fit_m_best])
        idx_keep = np.argsort(fit_b)[:pop_size]
        S_b = S_b[idx_keep]
        fit_b = fit_b[idx_keep]
        
        n_replace = max(1, int(b_n * pop_size))
        new_ab = np.random.uniform(lo, hi, (n_replace, 2))
        S_b[-n_replace:] = new_ab
        fit_b[-n_replace:] = fitness(new_ab)

        cur_best_idx = np.argmin(fit_b)
        cur_best_val = fit_b[cur_best_idx]
        cur_best_pos = S_b[cur_best_idx]

        history_best.append(float(cur_best_val))
        history_points.append([cur_best_pos[0], cur_best_pos[1]])

        
        if iteration % 5 == 0:
            swarm_snapshots.append({
                "xs": S_b[:, 0].tolist(),
                "ys": S_b[:, 1].tolist(),
                "gbest_x": float(cur_best_pos[0]),
                "gbest_y": float(cur_best_pos[1]),
                "gbest_val": float(cur_best_val),
                "iter": iteration,
            })

        
        if cur_best_val < best_val - 1e-10:
            best_val = cur_best_val
            stag_count = 0
        else:
            stag_count += 1
            if stag_count >= stagnation_limit:
                break

    best_idx = np.argmin(fit_b)
    best_pos = S_b[best_idx]

    return {
        "x": float(best_pos[0]),
        "y": float(best_pos[1]),
        "f": float(fit_b[best_idx]),
        "iterations": len(history_best),
        "history_best": history_best,
        "history_points": history_points,
        "swarm_snapshots": swarm_snapshots,
    }