from grad_calculus import gradient_descent_with_constant_step
from lr3.run import run_genetic_algorithm
import numpy as np

def run_hybrid_ga_gradient(
    f,
    grad,
    bounds,
    pop_size=100,
    n_gen=100,
    mut_rate=0.1,
    step0=0.001,
    eps1=1e-5,
    max_iter=10000
):
    ga_res = run_genetic_algorithm(
        f=f,
        bounds=bounds,
        pop_size=pop_size,
        n_gen=n_gen,
        mut_rate=mut_rate
    )



    x0, y0 = ga_res["best_x"], ga_res["best_y"]

    x_min, y_min, gd_iter, gd_path = gradient_descent_with_constant_step(
        f, grad,
        x0, y0,
        step0=step0,
        eps1=eps1,
        max_iter=max_iter
    )

    gd_points = np.array([[row[1], row[2]] for row in gd_path])

    return {
        "best_x": x_min,
        "best_y": y_min,
        "best_f": f(x_min, y_min),

        "ga_best_x": ga_res["best_x"],
        "ga_best_y": ga_res["best_y"],
        "ga_best_f": ga_res["best_f"],

        "ga_history_best": ga_res["history_best"],
        "ga_history_mean": ga_res["history_mean"],
        "ga_history_points": ga_res["history_points"],

        "gd_iterations": gd_iter,
        "gd_path": gd_points
    }