import numpy as np

def center_grad(f,x,y,h = 1e-5):
    df_dx = (f(x + h,y) - f(x - h,y)) / (2*h)
    df_dy = (f(x,y + h) - f(x,y-h)) / (2*h)
    return np.array([df_dx, df_dy])
def golden_section(f, a, b, eps = 1e-6):
    grad = (np.sqrt(5) - 1) / 2
    c = b - grad * (b-a)
    d = a + grad * (b-a)
    while abs(b - a) > eps:
        if f(c) < f(d): b = d
        else: a = c
        c = b - grad * (b-a)
        d = a + grad * (b -a)
    return (a+b)/2
