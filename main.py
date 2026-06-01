import json
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from lr2.main import QuadraticProblem, ModifiedSimplex
from lr3.run import run_genetic_algorithm
from lr4.main import run_pso
from lr5.run import run_bees_algorithm
from lr6.main import run_immune_network
from lr7.main import run_bacterial_optimization
from lr8.main import run_hybrid_ga_gradient
from numeric_algorithms import center_grad
from registry import FUNCTIONS, METHODS
from utils.animation_pso import create_swarm_animation

app = FastAPI()
templates = Jinja2Templates(directory="templates")


def create_optimization_chart(f, bounds, path_points, title="График"):
    x_range = np.linspace(bounds[0], bounds[1], 100)
    y_range = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f(X, Y)

    fig = go.Figure()
    fig.add_surface(x=X, y=Y, z=Z, opacity=0.8, colorscale='Viridis', showscale=False)

    pts = np.array(path_points)
    if pts.shape[1] == 2:
        zs = [f(p[0], p[1]) for p in pts]
        xs, ys = pts[:, 0], pts[:, 1]
    else:
        xs, ys, zs = pts[:, 1], pts[:, 2], pts[:, 3]

    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines+markers",
        marker=dict(size=4, color='red'),
        line=dict(color='white', width=2),
        name="Траектория"
    ))

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(aspectmode='cube'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=600
    )
    return fig.to_html(full_html=False)


@app.get("/", response_class=HTMLResponse)
async def index(
        request: Request,
        func: str = "Base function",
        method: str = "Gradient (const step)",
        x0: float = 2.0, y0: float = 2.0,
        step0: float = 0.01, eps1: float = 1e-5, max_iter: int = 1000
):
    meta = FUNCTIONS[func]
    f, bounds = meta["func"], meta["range"]

    x_min, y_min, iterations, path = METHODS[method](
        f, lambda x, y: center_grad(f, x, y),
        x0, y0, step0=step0, eps1=eps1, max_iter=max_iter
    )

    chart = create_optimization_chart(f, bounds, path)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "active_lab": 1,
        "graph": chart,
        "functions": FUNCTIONS.keys(),
        "results": {"x": round(x_min, 4), "y": round(y_min, 4), "f": round(f(x_min, y_min), 6), "k": iterations},
        "params": {"func": func, "x0": x0, "y0": y0, "step0": step0, "eps1": eps1}
    })


@app.get("/lab2", response_class=HTMLResponse)
async def lab2(request: Request, c_str: str = "-4, -6", D_str: str = "[[2, 1], [1, 2]]", A_str: str = "[[1, 2]]",
               b_str: str = "2"):
    try:
        c = np.array(json.loads(f"[{c_str}]"), dtype=float).flatten()
        D = np.array(json.loads(D_str), dtype=float)
        A = np.array(json.loads(A_str), dtype=float)
        b = np.array(json.loads(f"[{b_str}]"), dtype=float).flatten()

        qp = QuadraticProblem(c, D, A, b)
        M, B = qp.prepare_linear_system()
        M_aux, B_aux, c_aux = qp.build_auxiliary_lp(M, B)
        simplex = ModifiedSimplex(M_aux, B_aux, c_aux, qp.get_complementary_pairs())

        res = simplex.solve(verbose=False)
        if res:
            solution, obj_v = res
            x_opt = solution[:qp.n].tolist()
            f_opt = qp.objective_value(np.array(x_opt))
            status = "Успешно решено" if obj_v < 1e-6 else "Решение не найдено"
        else:
            x_opt, f_opt, status = None, None, "Метод не сошелся"
    except Exception as e:
        x_opt, f_opt, status = None, None, str(e)

    return templates.TemplateResponse("lab2.html", {
        "request": request,
        "active_lab": 2,
        "params": {"c": c_str, "D": D_str, "A": A_str, "b": b_str},
        "result": {"x": x_opt, "f": f_opt, "status": status}
    })


@app.get("/lab3", response_class=HTMLResponse)
async def lab3(request: Request, func: str = "Rosenbrock", pop_size: int = 100, n_gen: int = 100,
               mut_rate: float = 0.1):
    meta = FUNCTIONS[func]
    f, bounds = meta["func"], meta["range"]

    res = run_genetic_algorithm(f, bounds, pop_size, n_gen, mut_rate)

    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(y=res["history_best"], name="Best", line=dict(color='#38bdf8')))
    fig_conv.add_trace(go.Scatter(y=res["history_mean"], name="Mean", line=dict(color='gray', dash='dash')))
    fig_conv.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=30, b=10),
                           paper_bgcolor='rgba(0,0,0,0)')

    graph_surf = create_optimization_chart(f, bounds, res["history_points"])

    return templates.TemplateResponse("lab3.html", {
        "request": request,
        "active_lab": 3,
        "functions": FUNCTIONS.keys(),
        "params": {"func": func, "pop_size": pop_size, "n_gen": n_gen, "mut_rate": mut_rate},
        "result": res,
        "graph_conv": fig_conv.to_html(full_html=False),
        "graph_surf": graph_surf
    })

@app.get("/lab4", response_class=HTMLResponse)
async def lab4(
    request: Request,
    func: str = "Rastrigin",
    swarm_size: int = 50,
    n_iter: int = 200,
    current_ratio: float = 0.1,
    local_ratio: float = 1.0,
    global_ratio: float = 5.0,
):
    meta = FUNCTIONS[func]
    f, bounds = meta["func"], meta["range"]

    res = run_pso(f, bounds, swarm_size, n_iter,
                  current_ratio, local_ratio, global_ratio)

    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(
        y=res["history_best"],
        name="Лучшее значение",
        line=dict(color='#38bdf8')
    ))
    fig_conv.update_layout(
        template="plotly_dark", height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)'
    )

    graph_surf = create_optimization_chart(f, bounds, res["history_points"])

    graph_anim = create_swarm_animation(f, bounds, res["swarm_snapshots"])

    return templates.TemplateResponse("lab4.html", {
        "request": request,
        "active_lab": 4,
        "functions": FUNCTIONS.keys(),
        "params": {
            "func": func,
            "swarm_size": swarm_size,
            "n_iter": n_iter,
            "current_ratio": current_ratio,
            "local_ratio": local_ratio,
            "global_ratio": global_ratio,
        },
        "result": res,
        "graph_conv": fig_conv.to_html(full_html=False),
        "graph_surf": graph_surf,
        "graph_anim": graph_anim,
    })


@app.get("/lab5", response_class=HTMLResponse)
async def lab5(
        request: Request,
        func: str = "Rosenbrock",
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
):
    meta = FUNCTIONS[func]
    f, bounds = meta["func"], meta["range"]

    res = run_bees_algorithm(
        f, bounds,
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

    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(
        y=res["history_best"],
        name="Лучшее значение",
        line=dict(color='#f59e0b')
    ))
    fig_conv.update_layout(
        template="plotly_dark", height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)'
    )

    graph_surf = create_optimization_chart(f, bounds, res["history_points"])
    graph_anim = create_swarm_animation(f, bounds, res["swarm_snapshots"])

    return templates.TemplateResponse("lab5.html", {
        "request": request,
        "active_lab": 5,
        "functions": FUNCTIONS.keys(),
        "params": {
            "func": func,
            "n_scouts": n_scouts,
            "n_elite": n_elite,
            "n_perspective": n_perspective,
            "n_elite_bees": n_elite_bees,
            "n_persp_bees": n_persp_bees,
            "radius": radius,
            "max_iter": max_iter,
            "stagnation_limit": stagnation_limit,
            "use_iba": use_iba,
            "rho": rho,
        },
        "result": res,
        "graph_conv": fig_conv.to_html(full_html=False),
        "graph_surf": graph_surf,
        "graph_anim": graph_anim,
    })


@app.get("/lab6", response_class=HTMLResponse)
async def lab6(
        request: Request,
        func: str = "Rosenbrock",
        pop_size: int = 50,
        n_b: int = 10,
        n_c: int = 5,
        n_d: int = 5,
        alpha: float = 0.1,
        b_n: float = 0.2,
        max_iter: int = 300,
        stagnation_limit: int = 30,
):
    meta = FUNCTIONS[func]
    f, bounds = meta["func"], meta["range"]

    res = run_immune_network(
        f, bounds,
        pop_size=pop_size,
        n_b=n_b,
        n_c=n_c,
        n_d=n_d,
        alpha=alpha,
        b_n=b_n,
        max_iter=max_iter,
        stagnation_limit=stagnation_limit,
    )

    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(
        y=res["history_best"],
        name="Лучшее значение",
        line=dict(color='#10b981')
    ))
    fig_conv.update_layout(
        template="plotly_dark", height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)'
    )

    graph_surf = create_optimization_chart(f, bounds, res["history_points"])
    graph_anim = create_swarm_animation(f, bounds, res["swarm_snapshots"])

    return templates.TemplateResponse("lab6.html", {
        "request": request,
        "active_lab": 6,
        "functions": FUNCTIONS.keys(),
        "params": {
            "func": func,
            "pop_size": pop_size,
            "n_b": n_b,
            "n_c": n_c,
            "n_d": n_d,
            "alpha": alpha,
            "b_n": b_n,
            "max_iter": max_iter,
            "stagnation_limit": stagnation_limit,
        },
        "result": res,
        "graph_conv": fig_conv.to_html(full_html=False),
        "graph_surf": graph_surf,
        "graph_anim": graph_anim,
    })
@app.get("/lab7", response_class=HTMLResponse)
async def lab7(
    request: Request,
    func: str = "Sphere",
    pop_size: int = 20,
    n_chemotaxis: int = 50,
    n_reproduction: int = 4,
    n_elimination: int = 2,
    n_swim: int = 4,
    step_size: float = 0.1,
    p_eliminate: float = 0.25,
):
    meta = FUNCTIONS[func]
    f, bounds = meta["func"], meta["range"]

    res = run_bacterial_optimization(
        f=f,
        bounds=bounds,
        dims=2,
        pop_size=pop_size,
        n_chemotaxis=n_chemotaxis,
        n_reproduction=n_reproduction,
        n_elimination=n_elimination,
        n_swim=n_swim,
        step_size=step_size,
        p_eliminate=p_eliminate,
    )

    best_x, best_y = res["x"][0], res["x"][1]

    history_points = [
        [s["gbest_x"], s["gbest_y"]]
        for s in res["swarm_snapshots"]
    ]

    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(
        y=res["history_best"],
        name="Лучшее значение",
        line=dict(color='#f59e0b')
    ))
    fig_conv.update_layout(
        template="plotly_dark", height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)'
    )

    graph_surf = create_optimization_chart(f, bounds, history_points)
    graph_anim = create_swarm_animation(f, bounds, res["swarm_snapshots"])

    return templates.TemplateResponse("lab7.html", {
        "request": request,
        "active_lab": 7,
        "functions": FUNCTIONS.keys(),
        "params": {
            "func": func,
            "pop_size": pop_size,
            "n_chemotaxis": n_chemotaxis,
            "n_reproduction": n_reproduction,
            "n_elimination": n_elimination,
            "n_swim": n_swim,
            "step_size": step_size,
            "p_eliminate": p_eliminate,
        },
        "result": {**res, "x": best_x, "y": best_y},
        "graph_conv": fig_conv.to_html(full_html=False),
        "graph_surf": graph_surf,
        "graph_anim": graph_anim,
    })

@app.get("/lab8", response_class=HTMLResponse)
async def lab8(
    request: Request,
    func: str = "Rastrigin",
    pop_size: int = 100,
    n_gen: int = 100,
    mut_rate: float = 0.1,
    step0: float = 0.01,
    eps1: float = 1e-5,
    max_iter: int = 10000
):
    meta = FUNCTIONS[func]
    f, bounds = meta["func"], meta["range"]

    res = run_hybrid_ga_gradient(
        f,
        lambda x, y: center_grad(f, x, y),
        bounds,
        pop_size=pop_size,
        n_gen=n_gen,
        mut_rate=mut_rate,
        step0=step0,
        eps1=eps1,
        max_iter=max_iter
    )

    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(y=res["ga_history_best"], name="GA Best"))
    fig_conv.add_trace(go.Scatter(y=res["ga_history_mean"], name="GA Mean"))

    graph_ga = create_optimization_chart(f, bounds, res["ga_history_points"])
    graph_gd = create_optimization_chart(f, bounds, res["gd_path"])

    fig_conv.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return templates.TemplateResponse("lab8.html", {
        "request": request,
        "active_lab": 8,
        "functions": FUNCTIONS.keys(),
        "params": {
            "func": func,
            "pop_size": pop_size,
            "n_gen": n_gen,
            "mut_rate": mut_rate,
            "step0": step0,
            "eps1": eps1
        },
        "result": res,
        "graph_conv": fig_conv.to_html(full_html=False),
        "graph_ga": graph_ga,
        "graph_gd": graph_gd
    })