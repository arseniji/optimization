import numpy as np

class Particle:
    def __init__(self, swarm):
        self.position = np.random.uniform(swarm.min_vals, swarm.max_vals)
        self.velocity = np.random.uniform(
            swarm.min_vals - swarm.max_vals,
            swarm.max_vals - swarm.min_vals
        )
        self.best_position = self.position.copy()
        self.best_value = swarm.func(self.position)

    def update(self, swarm):
        phi = swarm.local_ratio + swarm.global_ratio
        chi = (2.0 * swarm.current_ratio /
               abs(2.0 - phi - np.sqrt(phi**2 - 4.0 * phi)))

        rp = np.random.rand(len(self.position))
        rg = np.random.rand(len(self.position))

        self.velocity = chi * (
            self.velocity
            + swarm.local_ratio * rp * (self.best_position - self.position)
            + swarm.global_ratio * rg * (swarm.best_position - self.position)
        )
        self.position += self.velocity

        val = swarm.func(self.position)
        if val < self.best_value:
            self.best_value = val
            self.best_position = self.position.copy()


class ParticleSwarm:
    def __init__(self, func, bounds, swarm_size=50,
                 current_ratio=0.1, local_ratio=1.0, global_ratio=5.0):
        assert local_ratio + global_ratio > 4, "local + global должно быть > 4"
        self.func = lambda pos: func(pos[0], pos[1])
        self.min_vals = np.array([bounds[0], bounds[0]])
        self.max_vals = np.array([bounds[1], bounds[1]])
        self.current_ratio = current_ratio
        self.local_ratio = local_ratio
        self.global_ratio = global_ratio

        self.particles = [Particle(self) for _ in range(swarm_size)]

        best = min(self.particles, key=lambda p: p.best_value)
        self.best_position = best.best_position.copy()
        self.best_value = best.best_value

    def next_iteration(self):
        for p in self.particles:
            p.update(self)
            if p.best_value < self.best_value:
                self.best_value = p.best_value
                self.best_position = p.best_position.copy()


def run_pso(func, bounds, swarm_size=50, n_iter=200,
            current_ratio=0.1, local_ratio=1.0, global_ratio=5.0):
    swarm = ParticleSwarm(func, bounds, swarm_size,
                          current_ratio, local_ratio, global_ratio)

    history_best = []
    history_points = []
    swarm_snapshots = [] 

    for i in range(n_iter):
        swarm.next_iteration()
        history_best.append(swarm.best_value)

        x, y = swarm.best_position
        history_points.append([x, y])

        if i % 5 == 0:
            snapshot = {
                "iter": i,
                "xs": [float(p.position[0]) for p in swarm.particles],
                "ys": [float(p.position[1]) for p in swarm.particles],
                "gbest_x": float(swarm.best_position[0]),
                "gbest_y": float(swarm.best_position[1]),
                "gbest_val": float(swarm.best_value),
            }
            swarm_snapshots.append(snapshot)

    return {
        "x": round(float(swarm.best_position[0]), 6),
        "y": round(float(swarm.best_position[1]), 6),
        "f": round(float(swarm.best_value), 8),
        "iterations": n_iter,
        "history_best": history_best,
        "history_points": history_points,
        "swarm_snapshots": swarm_snapshots,
    }