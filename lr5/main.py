import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Optional,Union


@dataclass
class BeeAlgorithmParams:
    n_scouts: int = 16          
    n_elite: int = 2            
    n_perspective: int = 3      
    n_elite_bees: int = 7       
    n_persp_bees: int = 4       
    radius: float = 0.2         
    max_iter: int = 500         
    stagnation_limit: int = 20  
    eps_elite: float = 1e-6     
    eps_persp: float = 1e-6
    use_iba: bool = False       
    rho: float = 0.9            


@dataclass
class BeesResult:
    x: float
    y: float
    f: float
    iterations: int
    history_best: List[float] = field(default_factory=list)
    history_points: List[Tuple] = field(default_factory=list)
    swarm_snapshots: List[dict] = field(default_factory=list)

class BeesAlgorithm:

    def __init__(
        self,
        func: Callable[[float, float], float],
        bounds,
        params: Optional[BeeAlgorithmParams] = None,
    ):
        self.func = func
        self.lo = np.array([bounds[0], bounds[0]])
        self.hi = np.array([bounds[1], bounds[1]])
        self.p = params or BeeAlgorithmParams()

        self._r = self.p.radius


    def _fitness(self, x: float, y: float) -> float:
        return -self.func(x, y)

    def _random_point(self) -> Tuple[float, float]:
        x = np.random.uniform(self.lo[0], self.hi[0])
        y = np.random.uniform(self.lo[1], self.hi[1])
        return float(x), float(y)

    def _clip(self, x: float, y: float) -> Tuple[float, float]:
        x = np.clip(x, self.lo[0], self.hi[0])
        y = np.clip(y, self.lo[1], self.hi[1])
        return float(x), float(y)

    def _random_in_patch(self, cx: float, cy: float, r: float) -> Tuple[float, float]:
        x = np.random.uniform(cx - r, cx + r)
        y = np.random.uniform(cy - r, cy + r)
        return self._clip(x, y)

    def _merge_patches(
        self,
        points: List[Tuple[float, float, float]],
        eps: float,
    ) -> List[Tuple[float, float, float]]:
        merged: List[Tuple[float, float, float]] = []
        used = [False] * len(points)
        for i, pi in enumerate(points):
            if used[i]:
                continue
            best = pi
            for j, pj in enumerate(points):
                if i == j or used[j]:
                    continue
                dist = np.sqrt((pi[0] - pj[0]) ** 2 + (pi[1] - pj[1]) ** 2)
                if dist <= eps:
                    used[j] = True
                    if pj[2] > best[2]:
                        best = pj
            merged.append(best)
        return merged

    def run(self) -> BeesResult:
        p = self.p
        self._r = p.radius

        history_best: List[float] = []
        history_points: List[Tuple] = []
        swarm_snapshots: List[np.ndarray] = []

        global_best_phi = -np.inf
        global_best_x = 0.0
        global_best_y = 0.0

        stagnation = 0
        prev_best_phi = -np.inf

        for t in range(p.max_iter):
            scouts = [self._random_point() for _ in range(p.n_scouts)]
            scout_data = [(x, y, self._fitness(x, y)) for x, y in scouts]

            scout_data.sort(key=lambda s: s[2], reverse=True)

            elite_centers = scout_data[: p.n_elite]
            persp_centers = scout_data[p.n_elite: p.n_elite + p.n_perspective]

            if p.eps_elite > 0:
                elite_centers = self._merge_patches(elite_centers, p.eps_elite)
            if p.eps_persp > 0:
                persp_centers = self._merge_patches(persp_centers, p.eps_persp)

            all_bees: List[Tuple[float, float, float]] = list(scout_data)

            for cx, cy, _ in elite_centers:
                for _ in range(p.n_elite_bees):
                    bx, by = self._random_in_patch(cx, cy, self._r)
                    all_bees.append((bx, by, self._fitness(bx, by)))

            for cx, cy, _ in persp_centers:
                for _ in range(p.n_persp_bees):
                    bx, by = self._random_in_patch(cx, cy, self._r)
                    all_bees.append((bx, by, self._fitness(bx, by)))

            all_bees.sort(key=lambda s: s[2], reverse=True)

            best_x, best_y, best_phi = all_bees[0]

            if best_phi > global_best_phi:
                global_best_phi = best_phi
                global_best_x = best_x
                global_best_y = best_y

            f_val = -global_best_phi
            history_best.append(f_val)
            history_points.append((t, global_best_x, global_best_y, f_val))

            snapshot = {
                "xs": [b[0] for b in all_bees],
                "ys": [b[1] for b in all_bees],
                "gbest_x": global_best_x,
                "gbest_y": global_best_y,
                "gbest_val": f_val,
                "iter": t,
            }
            if (t%5 == 0): swarm_snapshots.append(snapshot)

            if abs(best_phi - prev_best_phi) < 1e-10:
                stagnation += 1
            else:
                stagnation = 0
            prev_best_phi = best_phi

            if stagnation >= p.stagnation_limit:
                break
            if p.use_iba:
                if best_phi <= prev_best_phi:
                    self._r = min(self._r / p.rho, p.radius)
                else:
                    self._r = self._r * p.rho

        return BeesResult(
            x=global_best_x,
            y=global_best_y,
            f=-global_best_phi,
            iterations=len(history_best),
            history_best=history_best,
            history_points=history_points,
            swarm_snapshots=swarm_snapshots,
        )