from typing import Callable
from lr5.main import BeesAlgorithm, BeeAlgorithmParams, BeesResult


def run_bees_algorithm(
    func: Callable[[float, float], float],
    bounds: float,
    n_scouts: int = 16,
    n_elite: int = 2,
    n_perspective: int = 3,
    n_elite_bees: int = 7,
    n_persp_bees: int = 4,
    radius: float = 0.2,
    max_iter: int = 500,
    stagnation_limit: int = 20,
    use_iba: bool = False,
    rho: float = 0.9,
) -> dict:
    params = BeeAlgorithmParams(
        n_scouts=n_scouts,
        n_elite=n_elite,
        n_perspective=n_perspective,
        n_elite_bees=n_elite_bees,
        n_persp_bees=n_persp_bees,
        radius=radius,
        max_iter=max_iter,
        stagnation_limit=stagnation_limit,
        use_iba=use_iba,
        rho=rho,
    )

    algo = BeesAlgorithm(func, bounds, params)
    res: BeesResult = algo.run()

    return {
        "x": res.x,
        "y": res.y,
        "f": res.f,
        "iterations": res.iterations,
        "history_best": res.history_best,
        "history_points": res.history_points,
        "swarm_snapshots": res.swarm_snapshots,
    }