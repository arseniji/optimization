from functons import *
from grad_calculus import *

FUNCTIONS = {
    "Rastrigin": {
        "func": rastrigin,
        "range": (-5.12, 5.12),
        "center": (0, 0)
    },
    "Ackley": {
        "func": ackley,
        "range": (-5, 5),
        "center": (0, 0)
    },
    "Sphere": {
        "func": sphere,
        "range": (-5, 5),
        "center": (0, 0)
    },
    "Rosenbrock": {
        "func": rosenbrock,
        "range": (-2, 2),
        "center": (1, 1)
    },
    "Beale": {
        "func": bil,
        "range": (-4.5, 4.5),
        "center": (3, 0.5)
    },
    "Goldstein-Price": {
        "func": goldstein,
        "range": (-2, 2),
        "center": (0, -1)
    },
    "Booth": {
        "func": boot,
        "range": (-10, 10),
        "center": (1, 3)
    },
    "Bukin N6": {
        "func": bukin,
        "range": (-15, 5),
        "center": (-10, 1)
    },
    "Matyas": {
        "func": matias,
        "range": (-10, 10),
        "center": (0, 0)
    },
    "Levi N13": {
        "func": levi,
        "range": (-10, 10),
        "center": (1, 1)
    },
    "Himmelblau": {
        "func": himmelblau,
        "range": (-5, 5),
        "center": (3, 2)
    },
    "Three-hump camel": {
        "func": tree_hump_camel,
        "range": (-5, 5),
        "center": (0, 0)
    },
    "Easom": {
        "func": easom,
        "range": (-100, 100),
        "center": (np.pi, np.pi)
    },
    "Cross-in-tray": {
        "func": cross_in_tray,
        "range": (-10, 10),
        "center": (1.3491, -1.3491)
    },
    "Eggholder": {
        "func": eggholder,
        "range": (-512, 512),
        "center": (512, 404.2319)
    },
    "Table holder": {
        "func": table_holder,
        "range": (-10, 10),
        "center": (8.05502, 9.66459)
    },
    "Base function":{
        "func": base_function,
        "range": (-5,5),
        "center": (0,0)
    }
}



METHODS = {
    "Gradient (const step)": gradient_descent_with_constant_step,
}
